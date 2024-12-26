# %%
from typing import Union,Literal,AnyStr,Dict,List
import pandas as pd
import pathlib
import json
import chardet
from IPython.display import display, HTML
import re,json
# from pypdf import PdfReader
from PIL import Image as PILImage
import io
import dask.dataframe as dd
import multiprocessing
import torch
from transformers import AutoTokenizer
from datasets import IterableDataset, Dataset
import uuid
import pickle

current_file = pathlib.Path(__file__)
current_folder = current_file.parent
pypdf_corpus_parquet = current_folder/"pypdftexts.parquet"
corpus_parquet_pypdf_ocr_combined_path = current_folder/"ocr_all_texts.parquet"
num_workers = multiprocessing.cpu_count()
dask_compute_args = {'num_workers':num_workers,'scheduler':'threads'}
access_token = (current_folder/"access_token.txt").read_text(encoding="utf-8")

class GetdataClass():
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        # model_name = "meta-llama/Llama-3.2-1B-Instruct"
        # model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        # model_name = "facebook/MobileLLM-600M"
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True,
            token=access_token
        )

    def replace_spaces(self, text:Union[str,AnyStr,pd._libs.missing.NAType])->Union[str,AnyStr,pd._libs.missing.NAType]:
        if text is pd.NA:
            return text
        # Regular expression for traditional Chinese characters
        chinese_char_pattern = (
            r'[\u2E80-\u2EFF\u2F00-\u2FDF\u3000-\u303F\u3400-\u4DBF'
            r'\u4E00-\u9FA5\u9FA6-\u9FBB\u9FBC-\u9FC3\u9FC4-\u9FCB'
            r'\u9FCC-\u9FCC\u9FCD-\u9FD5\u9FD6-\u9FEA\u9FEB-\u9FEF'
            r'\u9FF0-\u9FFC\u9FFD-\u9FFF]'
        )

        # Regular expression to match spaces between traditional Chinese characters
        pattern = f'({chinese_char_pattern})\\s{{1,4}}({chinese_char_pattern})'

        # Replace spaces with an empty string
        return re.sub(pattern, r'\1\2', text)

    def remove_junk_chars(self, input:Union[str,AnyStr,pd._libs.missing.NAType])->Union[str,AnyStr,pd._libs.missing.NAType]:
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

    # 是I城也冇痂?

    def completedisplay(df):
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
            # display(HTML(df.to_html()))
            display(df)

    def extract_pdf_content(self, pdf_path)->Union[str,AnyStr,pd._libs.missing.NAType]:
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
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        try:
            page_contents = []
            # tessdata_dir_config = r'--tessdata-dir "/usr/share/tesseract-ocr/4.00/tessdata/"'
            for page_i,page in enumerate(reader.pages):
                text = page.extract_text().strip()
                if text!='':
                    text = f'<page number="{page_i}>{text}</page>"'
                    page_contents.append(text)
                # Extract images from the page
                # images = page.images
                # ocr_texts = []
                # for image_i, image in enumerate(images):
                #   try:
                #     image_pil = image.image
                #     ocr_text = pytesseract.image_to_string(image_pil, lang='chi_tra', config=tessdata_dir_config)
                #     ocr_text = replace_spaces(ocr_text)
                #     ocr_text = f"<ocr_text image_number="{image_i}">{ocr_text}</ocr_text>"
                #     ocr_texts.append(ocr_text)
                #   except Exception as e:
                #     print(f"Error processing image: {e}")
                # texts.extend(ocr_texts)
                # texts.append("")
                # page_contents.append("".join(texts))
            if len(page_contents)>0:
                return "".join(page_contents)
            else:
                return pd.NA
        except Exception as e:
            print(f"error at {pdf_path} for {e}")
            return pd.NA

    def get_corpus_df_with_pypdf(self, write:bool=False):
        if write:
            allpdf_files = (current_folder/"compset"/"reference"/"finance").glob("*.pdf")
            allpdf_files = list(allpdf_files)
            allpdf_files_series = pd.Series(allpdf_files)
            corpus_pypdf = allpdf_files_series.apply(lambda x: x.name).to_frame().rename(columns={0:'filename'})
            corpus_pypdf['category'] = corpus_pypdf['filename'].apply(lambda x: x.split("_")[0])
            corpus_pypdf = corpus_pypdf.reset_index(drop=True)
            corpus_pypdf['content'] = dd.from_map(
                    lambda x:pd.Series(
                        self.replace_spaces(
                            self.extract_pdf_content(x)
                        ), dtype='string[pyarrow]'),
                    allpdf_files,
                    meta = ('content','string[pyarrow]'),
                ).compute(**dask_compute_args).reset_index(drop=True)
            corpus_pypdf.to_parquet(pypdf_corpus_parquet)
        else:
            corpus_pypdf = pd.read_parquet(pypdf_corpus_parquet)
        return corpus_pypdf

    def convert_to_utf8(self, file_path):
        # Read the content of the file and detect its encoding
        try:
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
            
            # Convert the string from detected encoding to utf-8
            decoded_data = raw_data.decode(encoding)
            utf8_data = decoded_data.encode('utf-8')
            
            # Save the converted string back to the file path
            with open(file_path, 'wb') as file:
                file.write(utf8_data)
            return True
        except Exception as e:
            print(f"error at {file_path} for {e}")
            return False

    def corpus_from_pypdf_ocr_combined(self, directlyload:bool=True)->pd.DataFrame:
        # filename	category	content	content_len	doc_num
        if directlyload==False:
            df = self.get_corpus_df_with_pypdf(write=False)
            df['content_len'] = df['content'].apply(lambda x: 0 if x is pd.NA else len(x))
            filepath_series = df['filename'].apply(lambda x: (current_folder/'traindata_converted_txt'/x).with_suffix('.txt'))
            df['doc_num'] = df['filename'].apply(lambda x: int( (x.replace('.pdf','').split('_')[-1]) ) )
            df['category'] = df['category'].astype('category')
            not_sufficient_cond = ((df['content_len']<50) | (df['filename'].isin(['finance_803.pdf','finance_40.pdf']) ))
            df.loc[not_sufficient_cond,'content'] = filepath_series[not_sufficient_cond].apply(lambda x: self.remove_junk_chars( '<page number="0">'+pathlib.Path(x).read_text()+"</page>" )  )
            # df['content'] = df['content'].apply(remove_junk_chars)
            df['content_len'] = df['content'].apply(len)
            df.to_parquet(corpus_parquet_pypdf_ocr_combined_path)
            return df
        else:
            df = pd.read_parquet(corpus_parquet_pypdf_ocr_combined_path)
            return df

    def concatenate_chtext_list(self, input:List)->str:
        len_list = len(input)
        if len_list==1:
            return input[0]
        resultstr = input[0:len_list-1]
        resultstr = "、".join(resultstr)+f"及{input[-1]}"
        return resultstr        

    def get_corpus_df(self)->pd.DataFrame:
        dataset_folder = current_folder # /"compset"/"reference"/"faq"
        df_faqans = dataset_folder / "pid_map_content.json"
        df_faqans = df_faqans.read_text(encoding='utf-8')
        df_faqans = json.loads(df_faqans)
        df_faqans_list = []
        for key,rec in df_faqans.items():
            keyint = int(key)
            df_faqans_singledf = pd.DataFrame.from_records(rec).assign(faqid=keyint).assign(category='faq')
            df_faqans_list.append(df_faqans_singledf)

        df_faqans = pd.concat(df_faqans_list).reset_index(drop=True)
        df_faqans['answers'] = df_faqans['answers'].apply(lambda x: self.concatenate_chtext_list(x))
        # faqid	category	answers	question
        grouped_df_faqans = df_faqans.groupby(['faqid','category']).agg({
            'answers': lambda x: "答案："+"\n".join(x),
            'question': lambda x: '問題：'+"\n".join(x)
        }).reset_index()
        # filename	category	content	content_len	doc_num
        grouped_df_faqans = grouped_df_faqans.rename(columns={
            'question':'content',
            'faqid':'doc_num',
        })
        grouped_df_faqans['content'] = grouped_df_faqans['content']+grouped_df_faqans['answers']
        grouped_df_faqans = grouped_df_faqans.drop(columns=['answers'])
        grouped_df_faqans['content_len'] = grouped_df_faqans['content'].apply(lambda x: len(x))

        # ['filename', 'category', 'content', 'content_len', 'doc_num']
        # pdf_corpus = corpus_from_pypdf_ocr_combined(directlyload=True)
        from get_data_docling import docling_corpus_parquet
        pdf_corpus = pd.read_parquet(docling_corpus_parquet)
        returndf = pd.concat([pdf_corpus, grouped_df_faqans], axis=0).reset_index(drop=True)
        return returndf

    def apply_corpus_to_list_of_docnum(self, input:pd.Series, corpus:pd.DataFrame=None)->str:
        template = """<article id="{category}_{docnum}">{content}</article>"""
        subset_condition = (corpus['category']==input['category'])
        # category content doc_num
        resultstr = []
        for doc_num in input['source']:
            target_content = corpus['content'][subset_condition & (corpus['doc_num']==doc_num)]
            target_content = corpus['content'][subset_condition & (corpus['doc_num']==doc_num)].iat[0]
            # content_count = target_content.count()
            # if content_count<=0:
            #     print(f"error at {input['category']} {doc_num}")
            #     resultstr.append(str(doc_num)+',')
            # else:
            formatted_str = template.format(
                category=input['category'],
                docnum=doc_num,
                content=target_content
            )
            resultstr.append(formatted_str)
        resultstr = "".join(resultstr)
        return resultstr

    def get_competition_dataset_faq(self, join_corpus:bool=True, join_truths:bool=True, questions_filename:str="questions_example.json"):
        dataset_folder = current_folder #/"compset"/"dataset"/"preliminary"
        # qid	source	query	category
        df_questions = dataset_folder/questions_filename
        df_questions = df_questions.read_text(encoding='utf-8')
        df_questions = json.loads(df_questions)['questions']
        df_questions = pd.json_normalize(df_questions)
        df_questions['query'] = df_questions['query'].apply(lambda x: x.replace('「','『').replace('」','』') )
        # qid	retrieve	category
        if join_truths:
            df_truths = dataset_folder/"ground_truths_example.json"
            df_truths = df_truths.read_text(encoding='utf-8')
            df_truths = json.loads(df_truths)['ground_truths']
            df_truths = pd.json_normalize(df_truths)
            df_questions_and_answer = df_questions.merge(df_truths, how='inner', on=['qid','category'])
        else:
            df_questions_and_answer = df_questions

        if not join_corpus:
            return df_questions_and_answer

        corpus_df = self.get_corpus_df()
        df_questions_and_answer['context'] = df_questions_and_answer.apply(
            self.apply_corpus_to_list_of_docnum,
            axis=1,
            corpus=corpus_df
        )
        if join_truths:
            df_questions_and_answer['answer_in_articleid'] = df_questions_and_answer.apply(
                lambda x: x['category']+'_'+str(x['retrieve']),
                axis=1
            )
        df_questions_and_answer = df_questions_and_answer.drop(columns=['source','retrieve'], errors='ignore').rename(columns={'query':'question'})
        # columns: qid	question	category	context	answer_in_articleid
        return df_questions_and_answer

    # https://keras.io/examples/nlp/question_answering/
    # https://transformers.run/c3/2022-03-08-transformers-note-5/
    # https://huggingface.co/learn/nlp-course/en/chapter6/3b

    # %%


    # %%
    # Function to create the mask
    def create_mask(
            self,
            input_tensor:torch.Tensor,
            target_sequence:Union[int,torch.Tensor],
            return_mask:bool=False
        )-> Dict[str, Union[  List[Union[int,torch.Tensor]], torch.Tensor]  ]:
        # print(f"input_tensor {input_tensor} decoded as {self.tokenizer.decode(input_tensor)}")
        # print(f"target_sequence {target_sequence} decoded as {self.tokenizer.decode(target_sequence)}")
        target_len = target_sequence.size(-1)
        # Unfold the input tensor to create overlapping patches of the same size as the target sequence
        # like what a convolution windows does
        unfolded_tensor = input_tensor.unfold(dimension=-1, size=target_len, step=1)
        # Compare the unfolded patches with the target sequence
        matches = (unfolded_tensor == target_sequence).all(dim=-1)
        # Initialize a binary mask of the same shape as the input tensor
        if return_mask:
            mask_shape = list(input_tensor.shape)
            binary_mask = torch.zeros(mask_shape, dtype=torch.bool)
        mask_start_n = []
        mask_end_n = []
        mask_start_row_n = []
        for n_element in range(matches.size(-1)):
            selected_data = torch.select(matches,matches.dim()-1,n_element).squeeze()
            if selected_data.any():
                if selected_data.dim()==0:
                    mask_start_n.append(n_element)
                    mask_end_n.append(n_element+target_len)
                    mask_start_row_n.append(None)
                else:
                    non_zero_pos = torch.nonzero(selected_data, as_tuple=False).squeeze()
                    for pos in non_zero_pos:
                        if return_mask:
                            binary_mask[...,pos,n_element:n_element+target_len] = True
                        mask_start_n.append(n_element)
                        mask_end_n.append(n_element+target_len)
                        mask_start_row_n.append(pos)
        mask_start_n = [0] if len(mask_start_n)==0 else mask_start_n
        mask_end_n = [0] if len(mask_end_n)==0 else mask_end_n
        mask_start_row_n = [0] if len(mask_start_row_n)==0 else mask_start_row_n
        returndict = {'mask_start_n':mask_start_n,'mask_end_n':mask_end_n,'mask_start_row_n':mask_start_row_n,}
        if return_mask:
            returndict['binary_mask'] = binary_mask
        return returndict


    def markup_article_id(
            self,
            example: Dict[str, List], indices: List[int] = None, **kwargs
        ) -> Dict[str, List]:
        # context:Union[str,List[str]]=None, #必須已經包裝好
        # answer_in_articleid:Union[str,List[str]]=None,
        # question:Union[str,List[str]]=None,
        # answer_in_text:Union[str,List[str]]=None,
        # arg_max_length:int=None,
        context = None if ('context' not in example) else example['context']
        answer_in_articleid = None if ('answer_in_articleid' not in example) else example['answer_in_articleid']
        question = None if ('question' not in example) else example['question']
        answer_in_text = None if ('answer_in_text' not in example) else example['answer_in_text']
        arg_max_length = None if ('arg_max_length' not in example) else example['arg_max_length']
        
        if answer_in_articleid is None:
            answer_in_articleid = "art33884545"
        if context is None:
            context = """<article id="art66998787">股市熔斷機制（Stock Market Circuit Breaker）係指一種緊急使用的管制措施，常用於大盤及個股，旨在當股市發生劇烈波動時，透過強制暫停交易一段時間，讓投資人稍微冷靜，避免恐慌情緒蔓延，出現大規模賣壓。</article><article id="art33884545">本票係指民間企業為籌措短期資金或融通合法交易，所發行的一種短期票券，到期時發票人需支付票券所載之金額。</article>"""
        if question is None:
            question = "本票的概念是什麼?"
        if answer_in_text is None:
            task_is_qa = True
        else:
            task_is_qa = False
        qa_sep_token = self.tokenizer.convert_ids_to_tokens(8651)
        # self.tokenizer.sep_token = self.tokenizer.convert_ids_to_tokens(8651) # "||"

        input_is_list = True if isinstance(context, list) or isinstance(context, pd.Series) else False
        contexts = [context] if not input_is_list else context
        contexts = [f"<context>{c}</context>" for c in contexts]
        answers_in_articleid = [answer_in_articleid] if not input_is_list else answer_in_articleid
        questions = [question] if not input_is_list else question
        answers_in_text = [answer_in_text] if not input_is_list else answer_in_text
        # rag_docs = [{'title':answers_in_articleid[i],'text':c} for i, c in enumerate(contexts)] if input_is_list else []#{'art0':context}
        # rag_docs = [f"""<article id="art{answers_in_articleid[i]}">{c}</article>""" for i, c in enumerate(contexts)] # if not input_is_list else [f"""<context>{context}</context>"""]
        
        if task_is_qa:
            example_prompt = {
                "role": "user",
                "content": """
現在給你幾篇參考資料，不同的參考資料分別以<article id="資料編號">...</article>包裝並與其他參考資料分隔，你必須判斷使用者詢問的問題可以從哪一份參考資料中找到答案後回答該參考資料的資料編號（位於id=""中）。如果不存在可以回答問題的參考資料答0。回答時step by step思考推論。以下是範例
----範例開始----
問題：「本票的概念是什麼?」這個問題可以在什麼編號的參考資料中找到答案？
參考資料：<context>{septoken}<article id="art66998787">股市熔斷機制（Stock Market Circuit Breaker）係指一種緊急使用的管制措施，常用於大盤及個股，旨在當股市發生劇烈波動時，透過強制暫停交易一段時間，讓投資人稍微冷靜，避免恐慌情緒蔓延，出現大規模賣壓。</article><article id="art33884545">本票係指民間企業為籌措短期資金或融通合法交易，所發行的一種短期票券，到期時發票人需支付票券所載之金額。</article></context>
step by step推論：因為<article id="art33884545">...</article>的資料內有提到本票，而且文章編號為art33884545(id="art33884545")，所以正確回答是：art33884545
----範例結束----
請學習範例後準備回答。""".format(septoken=qa_sep_token).strip() #self.tokenizer.sep_token
            }
        else:
            example_prompt = {
                "role": "user",
                "content": """
現在給你幾篇參考資料，不同的參考資料分別以<article id="資料編號">...</article>包裝並與其他參考資料分隔，你必須根據使用者詢問的問題在參考之料中找到答案後回答問題。如果不存在可以回答問題的參考資料答0。回答時step by step思考推論。以下是範例
----範例開始----
問題：「本票的概念是什麼?」這個問題可以在什麼編號的參考資料中找到答案？
參考資料：<context>{septoken}<article id="art66998787">股市熔斷機制（Stock Market Circuit Breaker）係指一種緊急使用的管制措施，常用於大盤及個股，旨在當股市發生劇烈波動時，透過強制暫停交易一段時間，讓投資人稍微冷靜，避免恐慌情緒蔓延，出現大規模賣壓。</article><article id="art33884545">本票係指民間企業為籌措短期資金或融通合法交易，所發行的一種短期票券，到期時發票人需支付票券所載之金額。</article></context>
step by step推論：「本票的概念是什麼」和本票是什麼樣的事物、意義與內涵功用為何有相同意義，因為<article id="art33884545">...</article>的資料內有提到本票，而且提到本票「是民間企業為籌措短期資金或融通合法交易，所發行的一種短期票券」所以正確回答是：「民間企業為籌措短期資金或融通合法交易，所發行的一種短期票券」。
----範例結束----
請學習範例後準備回答。""".format(septoken=qa_sep_token).strip() #self.tokenizer.sep_token
            }

        prepare_prompt = [
                {"role": "system", "content": "你是一個擅長回答利用自己知識以及參考資料回答金融問題的專家助手。"},
                example_prompt,
                {"role": "assitant", "content":"了解，請提供給我問題和參考資料。"},
            ]
        prepare_prompt_message = self.tokenizer.apply_chat_template(
            prepare_prompt,
            tokenize=False,
            # chat_template='rag',
            add_generation_prompt=False,
            continue_final_message=False,
            # documents=rag_docs
        )
        
        if task_is_qa:
            user_prompts_question = [
                [
                    {
                        "role": "user",
                        "content": f"""問題：「{single_question}」這個問題可以在什麼編號的參考資料中找到答案？\n參考資料：<context>""".strip()
                    }
                ]
                for question_i, single_question in enumerate(questions)
            ]
        else:
            user_prompts_question = [
                [
                    {
                        "role": "user",
                        "content": f"""問題：{single_question}\n參考資料：""".strip()
                    }
                ]
                for question_i, single_question in enumerate(questions)
            ]
        # 參考資料：{contexts[question_i]}

        # messages = messages if input_is_list else messages[0]
        user_prompts_question_messages = self.tokenizer.apply_chat_template(
            user_prompts_question,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
            use_default_system_prompt=False
        )
        chat_systemprompt_begin_tokens = re.escape('<|begin_of_text|><|start_header_id|>system<|end_header_id|>')
        chat_systemprompt_end_tokens = re.escape('<|eot_id|>')
        replacement_system_prompt_str = fr"{chat_systemprompt_begin_tokens}\n\nCutting Knowledge Date: [A-Za-z]+\s\d\d\d\d\nToday\sDate:\s\d\d\s[A-Za-z\s]+\d\d\d\d\n\n{chat_systemprompt_end_tokens}"
        replacement_system_prompt_pattern = re.compile(replacement_system_prompt_str)
        user_prompts_question_messages = [replacement_system_prompt_pattern.sub('',t) for t in user_prompts_question_messages]

        tokenizer_general_kwargs = {
            'padding':'longest',
            'stride':30,
            'truncation':'only_second',
            'return_attention_mask':True,
            'return_overflowing_tokens':True,
            'return_tensors':'pt',
            'return_offsets_mapping':False,
            'add_special_tokens':False,
            'is_split_into_words':False,
        }
        tokenizer_target_kwargs = {
            'padding':False,
            'truncation':False,
            'return_attention_mask':False,
            'return_overflowing_tokens':False,
            'return_tensors':'pt',
            'return_offsets_mapping':False,
            'add_special_tokens':False,
            'is_split_into_words':False,
        }
        
        input_message_no_cut = [prepare_prompt_message+q for i, q in enumerate(user_prompts_question_messages)]
        if task_is_qa:
            #tokenizer_max_length = len(prepare_prompt_message)+50 if arg_max_length is None or arg_max_length==0 else arg_max_length
            self.tokenizer.pad_token = self.tokenizer.convert_ids_to_tokens(220) # it's ' '
            self.tokenizer.sep_token = qa_sep_token
            tokenizer_max_length = None
            inputs = self.tokenizer(
                text=input_message_no_cut,
                text_pair=contexts,
                max_length=tokenizer_max_length,
                **tokenizer_general_kwargs)
            # print(f"answers_in_articleid is {answers_in_articleid}")
            answers_tokens = self.tokenizer(
                answers_in_articleid,
                **tokenizer_target_kwargs
                )['input_ids']
            # print(f"answers_tokens is {answers_tokens}")
            inputs['start_positions'] = []
            inputs['end_positions'] = []
            # inputs['qaans_row_pos'] = []
        else:
            contexts_with_generation_prompts = [c+"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" for c in contexts]
            self.tokenizer.pad_token = self.tokenizer.eos_token
            tokenizer_max_length = None
            answers_in_dialogue = [
                [
                    {
                        "role": "assistant",
                        "content": f"""{single_answer}""".strip()
                    }
                ]
                for answer_i, single_answer in enumerate(answers_in_text)
            ]
            answers_chat_messages = self.tokenizer.apply_chat_template(
                answers_in_dialogue,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=False,
                use_default_system_prompt=False
            )
            answers_chat_messages = [replacement_system_prompt_pattern.sub('',m) for m in answers_chat_messages]
            
            inputs = self.tokenizer(
                text=input_message_no_cut,
                text_pair=contexts_with_generation_prompts,
                text_target=answers_chat_messages,
                max_length=tokenizer_max_length,
                **tokenizer_general_kwargs)
            answers_tokens = inputs['labels']

        answers_tokens_overflow_mapped = []        
        for i, correspond_i in enumerate(inputs["overflow_to_sample_mapping"]):
            input_ids = inputs["input_ids"][i]
            # bos_index = torch.where(input_ids==self.tokenizer.bos_token_id)
            if task_is_qa:
                # print(f"answers_tokens[correspond_i] is {answers_tokens[correspond_i]}")
                answer_pos_dict = self.create_mask(input_ids, answers_tokens[correspond_i], return_mask=False)
                min_start_n = min(answer_pos_dict['mask_start_n'])
                min_end_n = min(answer_pos_dict['mask_end_n'])
                inputs['start_positions'].append(min_start_n)
                inputs['end_positions'].append(min_end_n)
            else:
                # needs to be checked if inputs['labels'] is already answers_tokens_overflow_mapped
                answers_tokens_overflow_mapped.append(
                    answers_tokens[correspond_i]
                )
        
        if task_is_qa:
            # inputs['start_positions'] = torch.tensor(inputs['start_positions'])
            # inputs['end_positions'] = torch.tensor(inputs['end_positions'])
            pass
        else:
            inputs['labels'] = torch.cat(answers_tokens_overflow_mapped, axis=0).tolist()
        
        # for key in ['overflow_to_sample_mapping']: #'offset_mapping',
        #     inputs.pop(key)

        # print(f"input_is_list = {input_is_list}")
        for key in inputs:  
            if not input_is_list:
                inputs[key] = inputs[key][0]#.item()
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].tolist()
            # print(f"key={key}, inputs[key] = {inputs[key]}")


        return inputs

# %%
# Endless generator function
class EndlessGenerator:
    def __init__(self, df1, df2=None, generator_id=None):
        self.df1 = df1
        self.df2 = df2
        self.i = 0
        self.j = 0
        self.generator_id = (
            generator_id if generator_id is not None else str(uuid.uuid4())
        )
    def __call__(self):
        return self
    def __iter__(self):
        return self
    def __next__(self):
        # row_df1 = self.df1.iloc[self.i].to_dict()
        row_df1 = self.df1.iloc[self.i % len(self.df1.index)].to_dict()
        # row_df2 = self.df2.iloc[self.j % len(self.df2)].to_dict()
        # combined_row = {**row_df1, **row_df2}
        combined_row = row_df1
        self.i += 1
        # self.j += 1
        return combined_row
    def __len__(self):
        return len(self.df1.index)
    def __getitem__(self, index):
        # row_df1 = self.df1.iloc[index].to_dict()
        row_df1 = self.df1.iloc[index % len(self.df1.index)].to_dict()
        combined_row = row_df1
        return combined_row
    def __reduce__(self):
        return (self.__class__, (self.generator_id,) ) #(self.df1, self.df2), 

# %%
if __name__ == '__main__':
    # df = get_corpus_df_with_pypdf(write=True)
    # corpusdf = corpus_from_pypdf_ocr_combined(directlyload=False)
    

    # corpus_temp_df = get_corpus_df()
    # # corpus_pypdf = get_corpus_df_with_pypdf(write=False)
    # # corpus_from_pypdf_ocr_combined(True)#.dtypes
    # completedisplay(
    #     get_competition_dataset()
    #     # corpus_temp_df[(corpus_temp_df['category']=='insurance') & (corpus_temp_df['doc_num']==442)]
    #     # corpus_temp_df
    # )

    competition_dataframe = get_competition_dataset_faq()
    gen = EndlessGenerator(competition_dataframe)
    iterds = IterableDataset.from_generator(gen)
    iterds = iterds \
        .shuffle(seed=42, buffer_size=len(gen)) \
        .map(markup_article_id, batched=False, with_indices=False) \
        .remove_columns(column_names=['question','context'])
    # iterds = iterds.shuffle(seed=42)
    # https://huggingface.co/docs/datasets/about_mapstyle_vs_iterable
    for example_i, example in enumerate(iterds):
        print(example.keys())
        for key in example.keys():
            print(key)
            print(example[key])
        # print(example['answer_in_articleid'])
        # print( self.tokenizer.decode(
        #     example['input_ids'][example['start_positions']:example['end_positions']]
        # ))
        # print(example)
        break

    # ds = Dataset.from_dict(competition_dataframe.to_dict(orient='list'))
    # ds = ds.shuffle(seed=42)
    # ds = ds.map(markup_article_id, batched=False, with_indices=False, num_proc=multiprocessing.cpu_count())
    # display(ds[0])

    # markup_article_id(
    #     context=[
    #         """<article id="art202401160063">過去，美國聯邦準備體系的貼現窗口（Discount Window），主要係提供調整信用（Adjustment Credit）、擴大信用（Extended Credit）及季節性信用（Seasonal Credit）予存款貨幣機構。迨至2003年1月9日，聯邦準備理事會（FRB）核准12家聯邦準備銀行的要求，建立第一信用（Primary Credit）及第二信用（Secondary Credit）制度，分別取代原先的調整信用及擴大信用。 第一信用旨在幫助財務狀況健全的存款貨幣機構，應付暫時性之準備金需求，其中給予大銀行融通的期限大抵屬於隔夜性質；至於對小銀行的融通，一般亦在2個星期以內。第一信用利率（Primary Credit Rate）即為過去的重貼現率，起初為聯邦資金利率目標加一個百分點；迨至發生次級房貸危機，自2007年8月16日起縮小至0.5個百分點；2008年3月17日進一步縮小至0.25個百分點。第二信用則旨在提供信用給不符第一信用標準，且處在長期經營困難的存款貨幣機構；第二信用利率（Secondary Credit Rate）原則上係按第一信用利率加0.5個百分點。（詳細內容，請參考「國際金融新辭典」。）</article><article id="art66998787">股市熔斷機制（Stock Market Circuit Breaker）係指一種緊急使用的管制措施，常用於大盤及個股，旨在當股市發生劇烈波動時，透過強制暫停交易一段時間，讓投資人稍微冷靜，避免恐慌情緒蔓延，出現大規模賣壓。</article><article id="art33884545">本票係指民間企業為籌措短期資金或融通合法交易，所發行的一種短期票券，到期時發票人需支付票券所載之金額。""",
    #         """<article id="art202401160055">在第三版巴塞爾資本協定（Basel Accord III, Basel III）強調總體審慎監理改革下，為解決金融體系順循環問題，Basel III引進資本保留緩衝（Capital Conservation Buffer），要求銀行在最低資本適足率之上，須額外持有2.5%之資本，以普通股權益第一類資本支應，旨在確保銀行有額外可用的資本，俾在發生損失時可以提取。 BaselII的資本保留緩衝（Capital Conservation Buffer）旨在確保銀行有額外可用的資本，俾在發生損失時可以提取，於2016年起將逐年提列，於2019年達到2.5%最終要求。（詳細內容，請參考「國際金融新辭典」。）</article><article id="art66998787">股市熔斷機制（Stock Market Circuit Breaker）係指一種緊急使用的管制措施，常用於大盤及個股，旨在當股市發生劇烈波動時，透過強制暫停交易一段時間，讓投資人稍微冷靜，避免恐慌情緒蔓延，出現大規模賣壓。</article><article id="art33884545">本票係指民間企業為籌措短期資金或融通合法交易，所發行的一種短期票券，到期時發票人需支付票券所載之金額。""",
    #     ],
    #     # answer_in_articleid=["art66998787","art33884545"], #可用來測試答案重複出現在docs多處時怎麼處理
    #     # answer_in_articleid=["art202401160063","art202401160055"],
    #     question=["第一信用利率／第二信用利率是什麼?","第三版巴塞爾資本協定內涵是什麼?"],
    #     answer_in_text=["如說明1所示","如說明2所示"]
    # )