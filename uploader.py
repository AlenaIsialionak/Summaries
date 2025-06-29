import PyPDF2
import docx
import streamlit as st
from typing import Tuple, Union, Optional
import os

def read_file(filepath: str) -> Optional[str]:
    """
    Reads text from a file, automatically detecting the file format.

    :param filepath: str - The path to the file to be read.

    :return: Optional[str] - The text content of the file as a string, or None if the file
                              cannot be read or the format is unsupported.  Errors are displayed
                              using Streamlit's error message system.
    """
    
    try:
        filename, file_extension = os.path.splitext(filepath)
        file_extension = file_extension.lower()

        if file_extension == ".txt":
            return read_txt_file(filepath)
        elif file_extension == ".pdf":
            return read_pdf_file(filepath)
        elif file_extension == ".docx":
            return read_docx_file(filepath)
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None
    except Exception as e:
        st.error(f"Error reading file {filepath}: {e}")
        return None

def read_txt_file(filepath: str) -> Optional[str]:
    """
    Reads text from a .txt file.

    :param filepath: str - The path to the .txt file to be read.

    :return: Optional[str] - The text content of the file, or None if the file is not found
                              or an error occurs.  Errors are displayed using Streamlit's
                              error message system.  Uses UTF-8 encoding for proper character
                              handling.
    """

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
            return text
    except FileNotFoundError:
        st.error(f"The file {filepath} was not found.")
        return None
    except Exception as e:
        st.error(f"Error reading file {filepath}: {e}")
        return None


def read_pdf_file(filepath: str) -> Optional[str]:
    """
    Reads text from a .pdf file.

    :param filepath: str - The path to the .pdf file to be read.

    :return: Optional[str] - The text content of the file, or None if the file is not found
                              or an error occurs.  Errors are displayed using Streamlit's
                              error message system.  Requires the PyPDF2 library.
    """

    try:
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
            return text
    except FileNotFoundError:
        st.error(f"The file {filepath} was not found.")
        return None
    except Exception as e:
        st.error(f"Error reading file {filepath}: {e}")
        return None

def read_docx_file(filepath: str) -> Optional[str]:
    """
    Reads text from a .docx file.

    :param filepath: str - The path to the .docx file to be read.

    :return: Optional[str] - The text content of the file, or None if the file is not found
                              or an error occurs.  Errors are displayed using Streamlit's
                              error message system.  Requires the docx library.  Maintains
                              paragraph separation with newline characters.
    """

    try:
        doc = docx.Document(filepath)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except FileNotFoundError:
        st.error(f"The file {filepath} was not found.")
        return None
    except Exception as e:
        st.error(f"Error reading file {filepath}: {e}")
        return None