"""Unit tests for the elements module in the backend utils."""

import pytest
from pytest import FixtureRequest
from pytest_lazy_fixtures import lf

from backend.rag_components.elements import (
    Element,
    Image,
    Table,
    TableImage,
    TableText,
    Text,
)

# ----------------------------------- Text ----------------------------------- #


@pytest.fixture(
    params=[
        {"text": "Hello, World!", "format": "text"},
        {"text": "Hello, World!", "type": "text", "format": "text"},
        {"text": "Hello, World!", "format": "markdown", "metadata": {}},
        {
            "text": "Hello, World!",
            "format": "html",
            "metadata": {"metadata_1": "value_1"},
        },
    ]
)
def text_element(request: FixtureRequest) -> Text:
    """Provides different configurations of Text element."""
    return Text(**request.param)


def test_text(text_element: Text) -> None:
    """Test the Text element."""
    assert text_element.get_content() == text_element.text

    assert text_element.get_metadata() == {
        "type": "text",
        "format": text_element.format,
        **text_element.metadata,
    }


@pytest.mark.parametrize("type", ["text", "table", "image"])
@pytest.mark.parametrize("format", ["text", "html", "markdown", "image"])
def test_text_error(type: str, format: str) -> None:
    """Test the error cases for the Text element."""
    if type == "text" and format in ["text", "html", "markdown"]:
        Text(text="Hello, World!", format=format, type=type)
    else:
        with pytest.raises(ValueError):
            Text(text="Hello, World!", format=format, type=type)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"format": "text"},
        {"text": "Hello, World!"},
        {"text": "Hello, World!", "type": "text"},
        {"text": "Hello, World!", "format": "htm"},
        {"text": "", "format": "text"},
    ],
)
def test_text_error_2(kwargs: dict) -> None:
    """Test the error cases for the Text element."""
    with pytest.raises(ValueError):
        Text(**kwargs)


# ----------------------------------- Image ---------------------------------- #


@pytest.fixture(
    params=[
        {
            "base64": "base64_string_0",
            "mime_type": "image/jpeg",
        },
        {
            "base64": "base64_string_0",
            "mime_type": "image/jpeg",
            "type": "image",
        },
        {
            "base64": "base64_string_0",
            "mime_type": "image/jpeg",
            "format": "image",
        },
        {
            "base64": "base64_string_0",
            "mime_type": "image/jpeg",
            "type": "image",
            "format": "image",
        },
        {
            "base64": "base64_string_0",
            "mime_type": "image/jpeg",
            "metadata": {},
        },
        {
            "base64": "base64_string_1",
            "mime_type": "image/png",
            "metadata": {
                "metadata_1": "value_1",
            },
        },
    ]
)
def image_element(request: FixtureRequest) -> Image:
    """Provides different configurations of Image element."""
    return Image(**request.param)


def test_image(image_element: Image) -> None:
    """Test the Image element."""
    assert image_element.get_content() == image_element.base64

    assert image_element.get_metadata() == {
        "type": "image",
        "format": "image",
        "mime_type": image_element.mime_type,
        **image_element.metadata,
    }


@pytest.mark.parametrize("type", ["text", "image", "table"])
@pytest.mark.parametrize("format", ["text", "html", "markdown", "image"])
def test_image_error(type: str, format: str) -> None:
    """Test the error cases for the Image element."""
    if type == "image" and format == "image":
        Image(base64="base64_string", mime_type="image/jpeg", type=type, format=format)
    else:
        with pytest.raises(ValueError):
            Image(
                base64="base64_string", mime_type="image/jpeg", type=type, format=format
            )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mime_type": "image/jpeg"},
        {"base64": "base64_string"},
        {"base64": "base64_string", "mime_type": "image/jpe"},
        {"base64": "", "mime_type": "image/jpeg"},
    ],
)
def test_image_error_2(kwargs: dict) -> None:
    """Test the error cases for the Image element."""
    with pytest.raises(ValueError):
        Image(**kwargs)


# --------------------------------- TableText -------------------------------- #


@pytest.fixture(
    params=[
        {"text": "Hello, World!", "format": "text"},
        {"text": "Hello, World!", "type": "table", "format": "text"},
        {"text": "Hello, World!", "format": "markdown", "metadata": {}},
        {
            "text": "Hello, World!",
            "format": "html",
            "metadata": {"metadata_1": "value_1"},
        },
    ]
)
def tabletext_element(request: FixtureRequest) -> TableText:
    """Provides different configurations of TableText element."""
    return TableText(**request.param)


def test_tabletext(tabletext_element: TableText) -> None:
    """Test the TableText element."""
    assert tabletext_element.get_content() == tabletext_element.text

    assert tabletext_element.get_metadata() == {
        "type": "table",
        "format": tabletext_element.format,
        **tabletext_element.metadata,
    }


@pytest.mark.parametrize("type", ["text", "image", "table"])
@pytest.mark.parametrize("format", ["text", "html", "markdown", "image"])
def test_tabletext_error(type: str, format: str) -> None:
    """Test the error cases for the TableText element."""
    if type == "table" and format in ["text", "html", "markdown"]:
        TableText(text="Hello, World!", format=format, type=type)
    else:
        with pytest.raises(ValueError):
            TableText(text="Hello, World!", format=format, type=type)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"format": "text"},
        {"text": "Hello, World!"},
        {"text": "Hello, World!", "type": "table"},
        {"text": "Hello, World!", "format": "htm"},
        {"text": "", "format": "html"},
    ],
)
def test_tabletext_error_2(kwargs: dict) -> None:
    """Test the error cases for the TableText element."""
    with pytest.raises(ValueError):
        TableText(**kwargs)


# -------------------------------- TableImage -------------------------------- #


@pytest.fixture(
    params=[
        {
            "base64": "base64_string_0",
            "mime_type": "image/jpeg",
        },
        {
            "base64": "base64_string_0",
            "mime_type": "image/jpeg",
            "type": "table",
        },
        {
            "base64": "base64_string_0",
            "mime_type": "image/jpeg",
            "format": "image",
        },
        {
            "base64": "base64_string_0",
            "mime_type": "image/jpeg",
            "type": "table",
            "format": "image",
        },
        {
            "base64": "base64_string_0",
            "mime_type": "image/jpeg",
            "format": "image",
            "metadata": {},
        },
        {
            "base64": "base64_string_1",
            "mime_type": "image/png",
            "format": "image",
            "metadata": {
                "metadata_1": "value_1",
            },
        },
    ]
)
def tableimage_element(request: FixtureRequest) -> TableImage:
    """Provides different configurations of TableImage element."""
    return TableImage(**request.param)


def test_tableimage(tableimage_element: TableImage) -> None:
    """Test the TableImage element."""
    assert tableimage_element.get_content() == tableimage_element.base64

    assert tableimage_element.get_metadata() == {
        "type": "table",
        "format": tableimage_element.format,
        "mime_type": tableimage_element.mime_type,
        **tableimage_element.metadata,
    }


@pytest.mark.parametrize("type", ["text", "image", "table"])
@pytest.mark.parametrize("format", ["text", "html", "markdown", "image"])
def test_tableimage_error(type: str, format: str) -> None:
    """Test the error cases for the TableImage element."""
    if type == "table" and format == "image":
        TableImage(
            base64="base64_string", mime_type="image/jpeg", type=type, format=format
        )
    else:
        with pytest.raises(ValueError):
            TableImage(
                base64="base64_string", mime_type="image/jpeg", type=type, format=format
            )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mime_type": "image/jpeg"},
        {"base64": "base64_string"},
        {"base64": "base64_string", "mime_type": "image/jpe"},
        {"base64": "", "mime_type": "image/jpeg"},
    ],
)
def test_tableimage_error_2(kwargs: dict) -> None:
    """Test the error cases for the TableImage element."""
    with pytest.raises(ValueError):
        TableImage(**kwargs)


# ---------------------------------- Common ---------------------------------- #


@pytest.mark.parametrize(
    "element",
    [
        lf("text_element"),
        lf("image_element"),
        lf("tabletext_element"),
        lf("tableimage_element"),
    ],
)
def test_summary(element: Element) -> None:
    """Test the summary of the elements."""
    with pytest.raises(ValueError):
        _ = element.get_summary()

    element.set_summary("Summary")

    assert element.get_summary() == "Summary"

    with pytest.raises(ValueError):
        element.set_summary("")


def test_element_abstract_class() -> None:
    """Test the abstract class Element."""
    with pytest.raises(TypeError):
        _ = Element(type="text", format="text")


def test_table_abstract_class() -> None:
    """Test the abstract class Table."""
    with pytest.raises(TypeError):
        _ = Table(type="table", format="text")
