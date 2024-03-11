from typing import Any, Dict, Literal, Optional

from IPython.display import HTML, display
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
        self._display_content()
        self._display_summary()
        self._display_metadata()


class Text(Element):
    type: Literal["text"] = "text"
    text: str

    def get_content(self):
        return self.text

    def _display_content(self):
        display(HTML('<b style="color: red;">Text</b>'))
        print(self.text)


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
        display(HTML('<b style="color: red;">Image</b>'))
        display(HTML(f'<img src="data:{self.mime_type};base64,{self.base64}">'))


class Table(Element):
    type: Literal["table"] = "table"
    format: Literal["text", "html", "image"]

    def get_metadata(self):
        return {
            "type": self.type,
            "format": self.format,
            **self.metadata,
        }


class TableText(Table, Text):
    format: Literal["text", "html"]

    def _display_content(self):
        display(HTML('<b style="color: red;">Table</b>'))
        if self.format == "html":
            display(HTML(self.text))

        if self.format == "text":
            print(self.text)


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
                    case "text" | "html":
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
                    case _:
                        raise ValueError(f"Unsupported table format: {table_format}")
            case _:
                raise ValueError(f"Unsupported document type: {doc.metadata['type']}")
        elements.append(element)

    return elements
