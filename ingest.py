import glob
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from typing import List
import click
import torch
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from tqdm import tqdm

from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)

print("This is an adaptation of imartinez/privateGPT#560 to localGPT")

chunk_size = 100
chunk_overlap = 10


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in DOCUMENT_MAP:
        loader_class, loader_args = DOCUMENT_MAP[ext]
        loader = loader_class(file_path, **loader_args)
        
        return loader.load()


def load_document_batch(filepaths):
    logging.info("Loading document batch")
    with ThreadPoolExecutor(len(filepaths)) as exe:
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        data_list = [future.result() for future in futures]

        return (data_list, filepaths)


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:

    all_files = []
    for ext in DOCUMENT_MAP:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()

    return results


def split_documents(documents: list[Document]) -> tuple[list[Document], list[Document]]:
    # Splits documents for correct Text Splitter
    text_docs, python_docs = [], []
    for doc in documents:
        file_extension = os.path.splitext(doc.metadata["source"])[1]
        if file_extension == ".py":
            python_docs.append(doc)
        else:
            text_docs.append(doc)

    return text_docs, python_docs


@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)


def main(device_type):
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    text_documents, python_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=880, chunk_overlap=200
    )
    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))
    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")

    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device_type},
    )

    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )
    db.persist()
    db = None


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
