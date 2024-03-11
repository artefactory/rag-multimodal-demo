import base64
import tempfile
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from IPython.display import HTML, Markdown, display
from pydantic import BaseModel, PrivateAttr


class Element(BaseModel):
    type: Literal["text", "image", "table"]
    metadata: Dict[str, Any] = {}
    _summary: Optional[str] = PrivateAttr(None)

    def get_content(self):
        raise NotImplementedError

    def get_metadata(self):
        return {
            "type": self.type,
            **self.metadata,
        }

    def set_summary(self, summary: str):
        self._summary = summary

    def get_summary(self) -> str:
        if self._summary is None:
            raise ValueError("Summary not available")
        return self._summary

    def _display_content(self):
        raise NotImplementedError

    def _display_summary(self):
        if self._summary is not None:
            display(HTML('<b style="color: blue;">Summary</b>'))
            print(self._summary)

    def _display_metadata(self):
        metadata = self.get_metadata()
        display(HTML('<b style="color: blue;">Metadata</b>'))
        for key, value in metadata.items():
            print(f"  {key}: {value}")

    def _ipython_display_(self):
        class_name = self.__class__.__name__
        display(HTML(f'<b style="color: red;">{class_name}</b>'))

        self._display_content()
        self._display_summary()
        self._display_metadata()

    def export(self, folder_path: Path | str, filename: str):
        Path(folder_path).mkdir(parents=True, exist_ok=True)

        self._export_content(folder_path, filename)
        self._export_summary(folder_path, filename)

    def _export_content(self, folder_path: Path | str, filename: str):
        raise NotImplementedError

    def _export_summary(self, folder_path: Path | str, filename: str):
        if self._summary is not None:
            file_path = Path(folder_path) / f"{filename}.summary"
            file_path.write_text(self._summary)


class Text(Element):
    type: Literal["text"] = "text"
    text: str
    format: Literal["text", "html", "markdown"] = "text"

    def get_content(self):
        return self.text

    def get_metadata(self):
        return {
            "type": self.type,
            "format": self.format,
            **self.metadata,
        }

    def _display_content(self):
        match self.format:
            case "text":
                print(self.text)
            case "html":
                display(HTML(self.text))
            case "markdown":
                display(Markdown(self.text))
            case other:
                raise ValueError(f"Unsupported format: {other}")

    def _export_content(self, folder_path: Path | str, filename: str):
        match self.format:
            case "text":
                extension = "txt"
            case "html":
                extension = "html"
            case "markdown":
                extension = "md"
            case other:
                raise ValueError(f"Unsupported format: {other}")

        file_path = Path(folder_path) / f"{filename}.{extension}"
        file_path.write_text(self.text)


class Image(Element):
    type: Literal["image"] = "image"
    base64: str
    mime_type: Literal["image/jpeg", "image/png"]

    def get_content(self):
        return self.base64

    def get_metadata(self):
        return {
            "type": self.type,
            "mime_type": self.mime_type,
            **self.metadata,
        }

    def _display_content(self):
        display(HTML(f'<img src="data:{self.mime_type};base64,{self.base64}">'))

    def _export_content(self, folder_path: Path | str, filename: str):
        extension = self.mime_type.split("/")[1]
        file_path = Path(folder_path) / f"{filename}.{extension}"
        with file_path.open("wb") as file:
            file.write(base64.b64decode(self.base64))

    def get_local_path(self) -> Path:
        extension = self.mime_type.split("/")[1]
        # Create a temporary file to store the image and return the path
        with tempfile.NamedTemporaryFile(
            suffix=f".{extension}", delete=False
        ) as temp_file:
            temp_file.write(base64.b64decode(self.base64))
            return Path(temp_file.name)


class Table(Element):
    type: Literal["table"] = "table"
    format: Literal["text", "html", "markdown", "image"]

    def get_metadata(self):
        return {
            "type": self.type,
            "format": self.format,
            **self.metadata,
        }


class TableText(Table, Text):
    format: Literal["text", "html", "markdown"]


class TableImage(Table, Image):
    format: Literal["image"] = "image"


def langchain_doc_to_element(docs: list):
    elements = []
    for doc in docs:
        match doc.metadata["type"]:
            case "text":
                element = Text(text=doc.page_content, metadata=doc.metadata)
            case "image":
                element = Image(
                    base64=doc.page_content,
                    mime_type=doc.metadata["mime_type"],
                    metadata=doc.metadata,
                )

            case "table":
                table_format = doc.metadata["format"]
                match table_format:
                    case "text" | "html" | "markdown":
                        element = TableText(
                            text=doc.page_content,
                            format=table_format,
                            metadata=doc.metadata,
                        )

                    case "image":
                        element = TableImage(
                            base64=doc.page_content,
                            mime_type=doc.metadata["mime_type"],
                            metadata=doc.metadata,
                        )
                    case other:
                        raise ValueError(f"Unsupported table format: {other}")
            case other:
                raise ValueError(f"Unsupported document type: {other['type']}")
        elements.append(element)

    return elements
