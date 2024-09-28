import fitz
from pdf2image import convert_from_path
from difflib import SequenceMatcher
import tabula
import numpy as np
import cv2
from skimage.metrics import structural_similarity as compare_ssim


def pdf_text_similarity(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio()


def compare_text(pdf_path1, pdf_path2):
    doc1 = fitz.open(pdf_path1)
    doc2 = fitz.open(pdf_path2)

    for page_num in range(min(doc1.page_count, doc2.page_count)):
        page1_text = doc1[page_num].get_text()
        page2_text = doc2[page_num].get_text()

        similarity = pdf_text_similarity(page1_text, page2_text)
        print(f"Text Similarity (Page {page_num + 1}): {similarity}")

    doc1.close()
    doc2.close()


def compare_images(pdf_path1, pdf_path2):
    images1 = convert_from_path(pdf_path1)
    images2 = convert_from_path(pdf_path2)

    for i, (img1, img2) in enumerate(zip(images1, images2), 1):

        # Perform image comparison logic here
        # (e.g., using structural similarity index, pixel-wise comparison, etc.)
        # ...
        def compare_images(pdf_path1, pdf_path2):
            images1 = convert_from_path(pdf_path1)
            images2 = convert_from_path(pdf_path2)

            for i, (img1, img2) in enumerate(zip(images1, images2), 1):
                # Convert images to NumPy arrays
                img_np1 = np.array(img1)
                img_np2 = np.array(img2)

                # Resize images to the same dimensions
                img1_resized = cv2.resize(img_np1, (img_np2.shape[1], img_np2.shape[0]))

                # Compute Structural Similarity Index (SSI)
                ssi_index, _ = compare_ssim(img1_resized, img_np2, full=True)

                # You can set a threshold for similarity
                similarity_threshold = 0.95

                if ssi_index < similarity_threshold:
                    print(f"Difference found in image {i}")


def compare_tables(pdf_path1, pdf_path2):
    tables1 = tabula.read_pdf(pdf_path1, pages='all', multiple_tables=True)
    tables2 = tabula.read_pdf(pdf_path2, pages='all', multiple_tables=True)

    # Perform table comparison logic here
    # ...


if __name__ == "__main__":
    pdf_path1 = "C:\\Users\\MITYash\\PycharmProjects\\pdfComparator\\sample6.pdf"
    pdf_path2 = "C:\\Users\\MITYash\\PycharmProjects\\pdfComparator\\sample5.pdf"

    # Compare text
    compare_text(pdf_path1, pdf_path2)

    # Compare images
    compare_images(pdf_path1, pdf_path2)

    # Compare tables
    compare_tables(pdf_path1, pdf_path2)
