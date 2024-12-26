# %%
import sys,textwrap
from typing import Any, Dict, Final, List, Literal, Optional, Tuple, Union, AnyStr, Callable
import typing
import pandas as pd
import pathlib
import json,os
# import chardet
from IPython.display import display, HTML
import re,json
from PIL import Image as PILImage
import io
import dask.dataframe as dd
import multiprocessing

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import (
        DocumentConverter,
        PdfFormatOption,
    )
    from docling.pipeline.simple_pipeline import SimplePipeline
    from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
    from docling.datamodel.pipeline_options import (
        EasyOcrOptions,
        OcrOptions,
        PdfPipelineOptions,
    )
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling.backend.pdf_backend import PdfDocumentBackend
    from docling.utils.export import generate_multimodal_pages
    from docling.utils.utils import create_hash
    from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
    from docling_core.types.doc.labels import DocItemLabel, GroupLabel
    from docling_core.types.doc.base import ImageRefMode
    from docling_core.types.doc.document import DEFAULT_EXPORT_LABELS, ListItem, GroupItem, TextItem, DocItem, SectionHeaderItem, TableItem, PictureItem, ImageRef, NodeItem, DoclingDocument


    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.generate_page_images = False
    pipeline_options.generate_table_images = False
    pipeline_options.generate_picture_images = False
    pipeline_options.ocr_options = EasyOcrOptions(lang=['en','ch_tra'], use_gpu=True)

    doc_converter = (
        DocumentConverter(  # all of the below is optional, has internal defaults.
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.IMAGE,
            ],  # whitelist formats, non-matching files are ignored.
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options, # pipeline options go here.
                    backend=DoclingParseV2DocumentBackend # optional: pick an alternative backend
                    # PyPdfiumDocumentBackend
                ),
            },
        )
    )

    def iterate_items_apply_function(
        selfclass: DoclingDocument,
        root: Optional[NodeItem] = None,
        custom_func: Optional[Callable] = None,
        with_groups: bool = False,
        traverse_pictures: bool = True,
        page_no: Optional[int] = None,
        _level: int = 0,  # fixed parameter, carries through the node nesting level
    ) -> typing.Iterable[Tuple[NodeItem, int]]:  # tuple of node and level

        assert isinstance(selfclass, DoclingDocument)
        if isinstance(root, DoclingDocument):
            root = root.body

        if not isinstance(root, GroupItem) or with_groups:
            if hasattr(root, 'text') and custom_func is not None:
                root.text = custom_func(root.text)
            if isinstance(root, DocItem):
                if page_no is not None:
                    for pageno, prov in enumerate(root.prov):
                        print(f"level {_level} page {pageno} root type {type(root)}")
        # Traverse children
        for child_ref in root.children:
            child = child_ref.resolve(selfclass)
            if isinstance(child, NodeItem):
                # If the child is a NodeItem, recursively traverse it
                if not isinstance(child, PictureItem) or traverse_pictures:
                    iterate_items_apply_function(
                        selfclass,
                        child, 
                        custom_func = custom_func,
                        _level=_level + 1, with_groups=with_groups,
                    )



    def extract_pdf_content(source)->Union[str,AnyStr,pd._libs.missing.NAType]:
        """
        Extracts text and OCR-processed text from a PDF file.

        Args:
        pdf_path: Path to the PDF file.

        Returns:
        A list of strings, where each element is the combined text and OCR-text of a page.
        """
        # try:
        #     import pytesseract
        # except:
        #     pass
        try:
            conv_res = doc_converter.convert(source)
            iterate_items_apply_function(conv_res, conv_res, replace_spaces)
            mdresult = conv_res.document.export_to_markdown()
            if len(mdresult)<=0:
                return pd.NA
            print(f"source: {source} done.")
            return mdresult
        except Exception as e:
            print(f"error at {source} for {e}")
            return pd.NA

    def get_corpus_df_with_docling(write:bool=False):
        global doc_converter
        if write:
            allpdf_files = (current_folder/"compset"/"reference"/"finance").glob("*.pdf")
            allpdf_files = list(allpdf_files)
            allpdf_files_series = pd.Series(allpdf_files)
            corpus_from_pdf_df = allpdf_files_series.apply(lambda x: x.name).to_frame().rename(columns={0:'filename'})
            corpus_from_pdf_df['category'] = corpus_from_pdf_df['filename'].apply(lambda x: x.split("_")[0])
            corpus_from_pdf_df = corpus_from_pdf_df.reset_index(drop=True)
            # corpus_from_pdf_df['content'] = dd.from_map(
            #         lambda x:pd.Series(extract_pdf_content(x), dtype='string[pyarrow]'),
            #         allpdf_files,
            #         meta = ('content','string[pyarrow]'),
            #     ).compute(**dask_compute_args).reset_index(drop=True)
            # corpus_from_pdf_df['content'] = allpdf_files_series.apply(extract_pdf_content)
            contents_list = []
            batch_size = 20
            for i in range(0,len(allpdf_files),batch_size):
                end_i = i+batch_size if (i+batch_size)<=len(allpdf_files) else len(allpdf_files)
                conv_results_iter = doc_converter.convert_all(allpdf_files[i:end_i], raises_on_error=False) # previously `convert`
                conv_results_iter = list(conv_results_iter)
                for iteri, conv_result in enumerate(conv_results_iter):
                    try:
                        iterate_items_apply_function(conv_results_iter[iteri].document, conv_results_iter[iteri].document, lambda x:remove_junk_chars(replace_spaces(x)) )
                        conv_results_iter[iteri] = conv_results_iter[iteri].document.export_to_markdown()
                    except Exception as e:
                        print(f"error at {allpdf_files[i+iteri]} for {e}")
                        conv_results_iter[iteri] = pd.NA
                contents_list.extend(conv_results_iter)
            corpus_from_pdf_df['content'] = pd.Series(contents_list)
            corpus_from_pdf_df['content_len'] = corpus_from_pdf_df['content'].apply(lambda x: 0 if x is pd.NA else len(x))
            corpus_from_pdf_df['doc_num'] = corpus_from_pdf_df['filename'].apply(lambda x: int(x.replace('.pdf','').split('_')[1]))
            corpus_from_pdf_df.to_parquet(docling_corpus_parquet)
        else:
            corpus_from_pdf_df = pd.read_parquet(docling_corpus_parquet)
        return corpus_from_pdf_df

    docling_imported = True
except:
    docling_imported = False

import multiprocessing

os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())


current_file = pathlib.Path(__file__)
current_folder = current_file.parent
docling_corpus_parquet = current_folder/"doclingtexts.parquet"
corpus_parquet_pypdf_ocr_combined_path = current_folder/"ocr_all_texts.parquet"
num_workers = multiprocessing.cpu_count()
dask_compute_args = {'num_workers':num_workers,'scheduler':'threads'}
access_token = (current_folder/"access_token.txt").read_text(encoding="utf-8")
chinese_char_pattern = r'[\u4E00-\u9FFF\u3400-\u4DBF\U00020000-\U0002A6DF\U0002A700-\U0002EBEF\U00030000-\U000323AF\uFA0E-\uFA29\u3006\u3007，。！？：；「」『』（）［］｛｝《》〈〉﹏﹋﹌﹏、‧・。～—…．]'
replace_spaced_cht_characters_pattern = fr"({chinese_char_pattern})\s{{1,4}}({chinese_char_pattern})"
replace_spaced_cht_characters_pattern = re.compile(replace_spaced_cht_characters_pattern)
replace_empty_lines_pattern = re.compile(r'^\s*$\n', re.MULTILINE)

def replace_spaces(
        input:Union[str,AnyStr,pd._libs.missing.NAType]
    )->Union[str,AnyStr,pd._libs.missing.NAType]:
    if input is pd.NA:
        return input
    global replace_spaced_cht_characters_pattern
    global replace_empty_lines_pattern
    # Regular expression for traditional Chinese characters
    # chinese_char_pattern = (
    #     r'[\u2E80-\u2EFF\u2F00-\u2FDF\u3000-\u303F\u3400-\u4DBF'
    #     r'\u4E00-\u9FA5\u9FA6-\u9FBB\u9FBC-\u9FC3\u9FC4-\u9FCB'
    #     r'\u9FCC-\u9FCC\u9FCD-\u9FD5\u9FD6-\u9FEA\u9FEB-\u9FEF'
    #     r'\u9FF0-\u9FFC\u9FFD-\u9FFF]'
    # )
    # chinese_char_pattern = r'[\u4E00-\u9FFF\u3400-\u4DBF\U00020000-\U0002A6DF\U0002A700-\U0002EBEF\U00030000-\U000323AF\UFA0E-\UFA29\u3006\u3007]'
    # chinese_char_pattern = r'[\u4E00-\u9FFF\u3400-\u4DBF\U00020000-\U0002A6DF\U0002A700-\U0002EBEF\U00030000-\U000323AF\uFA0E-\uFA29\u3006\u3007]'

    # Regular expression to match spaces between traditional Chinese characters
    # pattern = fr"({chinese_char_pattern})\s{{1,4}}({chinese_char_pattern})"
    # pattern = re.compile(pattern)

    # Replace spaces with an empty string
    input = replace_empty_lines_pattern.sub('', input)
    input = input.replace(" ( ","（").replace(" ) ","）").replace(" / ","／")
    while True:
        output = replace_spaced_cht_characters_pattern.sub(r'\1\2', input)
        output = output.strip()
        if output == input:
            break
        input = output
    return output


def remove_junk_chars(input:Union[str,AnyStr,pd._libs.missing.NAType])->Union[str,AnyStr,pd._libs.missing.NAType]:
    if input is pd.NA:
        return input
    junkchars = """皿屮彳•«口二匚^一|/=■—I|^—”■■ҽԖज़ϦљϷηϦљӝ଄!୍ൔ""".strip().replace("\n","")
    junkwords = ['f?','Illi','??','****','***','**','0p','i~','r?','....','T^']
    iter_i = 0
    while iter_i < len(junkchars):
        junkwords.append(junkchars[iter_i:iter_i+1])
        iter_i += 1
    for word in junkwords:
        input = input.replace(word,'')
    output = input
    return output

def completedisplay(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        # display(HTML(df.to_html()))
        display(df)


# %%
if __name__ == '__main__':
    # %%
    print("MAIN")
    df = get_corpus_df_with_docling(write=True)
    # completedisplay(df)
    # %%
