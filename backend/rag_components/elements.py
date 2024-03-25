"""Classes for representing multimodal elements with a type, format, and metadata."""

import base64
import tempfile
from abc import abstractmethod
from pathlib import Path
from typing import Any, Literal

from IPython.display import HTML, Markdown, display
from langchain_core.documents import Document
from pydantic import BaseModel, PrivateAttr, validator

from backend.utils.image import local_image_to_base64


class Element(BaseModel):
    """Abstract base class representing an element with a type, format, and metadata.

    Attributes:
        type ("text", "image", "table"): Type of the element.
        format ("text", "html", "markdown", "image"): Format of the element's content.
        metadata (dict[str, Any]): Additional metadata for the element.
        _summary (str | None): Private attribute for storing a summary of the element.

    Methods:
        get_content: Get the content of the element.
        get_metadata: Returns the metadata of the element.
        set_summary: Sets the summary of the element.
        get_summary: Gets the summary of the element.
        _display_content: Abstract method to display the content of the element.
        _display_summary: Displays the summary of the element.
        _display_metadata: Displays the metadata of the element.
        _ipython_display_: Displays the element in IPython environments.
        export: Exports the content and summary of the element to a specified folder.
        _export_content: Exports the content of the element.
        _export_summary: Exports the summary of the element.
    """

    type: Literal["text", "image", "table"]
    format: Literal["text", "html", "markdown", "image"]
    metadata: dict[str, Any] = {}
    _summary: str | None = PrivateAttr(None)

    @abstractmethod
    def get_content(self) -> str:
        """Abstract method to get the content of the element."""
        pass

    def get_metadata(self) -> dict[str, Any]:
        """Returns the metadata of the element."""
        return {
            "type": self.type,
            "format": self.format,
            **self.metadata,
        }

    def set_summary(self, summary: str) -> None:
        """Sets the summary of the element.

        Args:
            summary (str): The summary to set.
        """
        summary = validate_string(summary)
        self._summary = summary

    def get_summary(self) -> str:
        """Gets the summary of the element.

        Raises:
            ValueError: If the summary is not set.

        Returns:
            str: The summary of the element.
        """
        if self._summary is None:
            raise ValueError("Summary not available")
        return self._summary

    @abstractmethod
    def _display_content(self) -> None:
        """Abstract method to display the content of the element."""
        pass

    def _display_summary(self) -> None:
        """Displays the summary of the element, if available."""
        if self._summary is not None:
            display(HTML('<b style="color: blue;">Summary</b>'))
            print(self._summary)

    def _display_metadata(self) -> None:
        """Displays the metadata of the element."""
        metadata = self.get_metadata()
        display(HTML('<b style="color: blue;">Metadata</b>'))
        for key, value in metadata.items():
            print(f"  {key}: {value}")

    def _ipython_display_(self) -> None:
        """Displays the element in IPython environments."""
        class_name = self.__class__.__name__
        display(HTML(f'<b style="color: red;">{class_name}</b>'))

        self._display_content()
        self._display_summary()
        self._display_metadata()

    def export(self, folder_path: Path | str, filename: str) -> None:
        """Exports the content and summary of the element to a specified folder.

        Args:
            folder_path (Path | str): Folder path to export the content and summary to.
            filename (str): Filename (without extension) to use for the exported files.
        """
        Path(folder_path).mkdir(parents=True, exist_ok=True)

        self._export_content(folder_path, filename)
        self._export_summary(folder_path, filename)

    @abstractmethod
    def _export_content(self, folder_path: Path | str, filename: str) -> None:
        """Abstract method to export the content of the element."""
        pass

    def _export_summary(self, folder_path: Path | str, filename: str) -> None:
        """Exports the summary of the element, if available."""
        if self._summary is not None:
            file_path = Path(folder_path) / f"{filename}.summary"
            file_path.write_text(self._summary)

    @validator("*", pre=True)
    def validate_string_field(cls, value: Any) -> Any:  # noqa: ANN401
        """Validates that the string fields are not empty.

        Args:
            value (Any): Value of the field to validate.

        Returns:
            Any: Validated value.
        """
        if isinstance(value, str):
            return validate_string(value)
        return value


def validate_string(value: str) -> str:
    """Validates that a string is not empty.

    Args:
        value (str): String to validate.

    Raises:
        ValueError: If the string is empty.

    Returns:
        str: Validated string.
    """
    if not value.strip():
        raise ValueError("String cannot be empty")
    return value


class Text(Element):
    """Class representing a text element.

    Inherits from Element and adds specific attributes and methods for text content.

    Attributes:
        type ("text"): The type of the element, set to "text".
        format ("text", "html", "markdown"): The format of the text content.
        text (str): The actual text content.

    Methods:
        get_content: Returns the text content.
        _display_content: Displays the text content in the specified format.
        _export_content: Exports the text to a file with the appropriate extension.
    """

    type: Literal["text"] = "text"
    format: Literal["text", "html", "markdown"]
    text: str

    def get_content(self) -> str:
        """Returns the text content."""
        return self.text

    def _display_content(self) -> None:
        """Displays the text content in the appropriate format."""
        match self.format:
            case "text":
                print(self.text)
            case "html":
                display(HTML(self.text))
            case "markdown":
                display(Markdown(self.text))
            case other:
                raise ValueError(f"Unsupported format: {other}")

    def _export_content(self, folder_path: Path | str, filename: str) -> None:
        """Exports the text content to a file with the appropriate extension.

        Args:
            folder_path (Path | str): Folder path to export the content to.
            filename (str): Filename (without extension) to use for the exported file.

        Raises:
            ValueError: If the format is not supported.
        """
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
    """Class representing an image element.

    Inherits from Element and adds specific attributes and methods for image content.

    Attributes:
        type ("image"): The type of the element, set to "image".
        format ("image"): The format of the element's content, set to "image".
        mime_type ("image/jpeg", "image/png"): The MIME type of the image.
        base64 (str): The base64 encoded string of the image content.

    Methods:
        get_content: Returns the base64 encoded string of the image.
        get_metadata: Returns the metadata of the image, including the MIME type.
        _display_content: Displays the image in an IPython environment.
        _export_content: Exports the image to a file with the appropriate extension.
        get_local_path: Creates a temporary file with the image and returns the path.
    """

    type: Literal["image"] = "image"
    format: Literal["image"] = "image"
    mime_type: Literal["image/jpeg", "image/png"]
    base64: str

    def get_content(self) -> str:
        """Returns the base64 encoded string of the image."""
        return self.base64

    def get_metadata(self) -> dict[str, Any]:
        """Returns the metadata of the image, including the MIME type."""
        return {
            "type": self.type,
            "format": self.format,
            "mime_type": self.mime_type,
            **self.metadata,
        }

    def _display_content(self) -> None:
        """Displays the image in an IPython environment."""
        display(HTML(f'<img src="data:{self.mime_type};base64,{self.base64}">'))

    def _export_content(self, folder_path: Path | str, filename: str) -> None:
        """Exports the image to a file with the appropriate extension.

        Args:
            folder_path (Path | str): Folder path to export the image to.
            filename (str): Filename (without extension) to use for the exported file.
        """
        extension = self.mime_type.split("/")[1]
        file_path = Path(folder_path) / f"{filename}.{extension}"
        with file_path.open("wb") as file:
            file.write(base64.b64decode(self.base64))

    def get_local_path(self) -> Path:
        """Creates a temporary file with the image and returns the path.

        Returns:
            Path: The path to the temporary file.
        """
        extension = self.mime_type.split("/")[1]
        # Create a temporary file to store the image and return the path
        with tempfile.NamedTemporaryFile(
            suffix=f".{extension}", delete=False
        ) as temp_file:
            temp_file.write(base64.b64decode(self.base64))
            return Path(temp_file.name)


class Table(Element):
    """Abstract class representing a table element.

    Inherits from Element and serves as a base for table-related elements.
    """

    type: Literal["table"] = "table"


class TableText(Table, Text):
    """Class representing a table with text content.

    Inherits from Table and Text, combining their attributes and methods.
    """

    format: Literal["text", "html", "markdown"]


class TableImage(Table, Image):
    """Class representing a table with image content.

    Inherits from Table and Image, combining their attributes and methods.
    """

    format: Literal["image"] = "image"


def _create_text_element(doc: Document, element_class: type[Text]) -> Text:
    """Create a text element from a Langchain Document object.

    Args:
        doc (Document): Langchain Document object.
        element_class (Type[Text]): Text element class to create (Text, TableText).

    Returns:
        Element: Text element created from the Document object.
    """
    source = doc.metadata.get("source", "content")
    match source:
        case "content":
            element = element_class(
                type=doc.metadata["type"],
                format=doc.metadata["format"],
                text=doc.page_content,
                metadata=doc.metadata,
            )
        case "summary":
            element = element_class(
                type=doc.metadata["type"],
                format=doc.metadata["format"],
                text="No content available",
                metadata=doc.metadata,
            )
            element.set_summary(doc.page_content)
        case other:
            raise ValueError(f"Unsupported element source: {other}")
    return element


NO_IMAGE = local_image_to_base64("img/no_image.png")


def _create_image_element(doc: Document, element_class: type[Image]) -> Image:
    """Create an image element from a Langchain Document object.

    Args:
        doc (Document): Langchain Document object.
        element_class (Type[Image]): Image element class to create (Image, TableImage).

    Returns:
        Element: Image element created from the Document object.
    """
    source = doc.metadata.get("source", "content")
    match source:
        case "content":
            element = element_class(
                type=doc.metadata["type"],
                format=doc.metadata["format"],
                base64=doc.page_content,
                mime_type=doc.metadata["mime_type"],
                metadata=doc.metadata,
            )
        case "summary":
            element = element_class(
                type=doc.metadata["type"],
                format=doc.metadata["format"],
                base64=NO_IMAGE,
                mime_type="image/png",
                metadata=doc.metadata,
            )
            element.set_summary(doc.page_content)
        case other:
            raise ValueError(f"Unsupported element source: {other}")
    return element


def convert_documents_to_elements(docs: list[Document]) -> list:
    """Convert a list of Langchain Document objects to a list of Element objects.

    Args:
        docs (list[Document]): List of Document objects to convert.

    Raises:
        ValueError: If the document type or format is not supported.

    Returns:
        list: List of Element objects.
    """
    elements = []
    for doc in docs:
        match doc.metadata["type"]:
            case "text":
                element = _create_text_element(doc, Text)
            case "image":
                element = _create_image_element(doc, Image)
            case "table":
                match doc.metadata["format"]:
                    case "text" | "html" | "markdown":
                        element = _create_text_element(doc, TableText)
                    case "image":
                        element = _create_image_element(doc, TableImage)
                    case other:
                        raise ValueError(f"Unsupported table format: {other}")
            case other:
                raise ValueError(f"Unsupported document type: {other}")
        elements.append(element)

    return elements
