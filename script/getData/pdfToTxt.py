# Text still gets cleaned manually. If text data is available, there's no need to run this file.
import io
import os
import re
from collections import deque
from pdfminer.layout import LAParams, LTTextBox
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator, TextConverter
from nltk.tokenize import sent_tokenize

path = os.path.dirname(os.getcwd())
pdf_path = os.path.join(path, "../data_pdf/")
txt_path = os.path.join(path, "../txtData_b/")

def removeNotAscii(input):
    return input.encode('ascii','ignore').decode()  

def prepare_data(pdf_file, txt_file): #convert and fetch headers
    rsrcmgr = PDFResourceManager()
#stuff for fetching headers
    device = PDFPageAggregator(rsrcmgr, laparams=LAParams())
    interpreter_fetch = PDFPageInterpreter(rsrcmgr, device) 
#stuff for converting files
    file_handler = io.StringIO()
    converter = TextConverter(rsrcmgr, file_handler)
    interpreter_convert = PDFPageInterpreter(rsrcmgr, converter)
    headers = []
    
    of_pdf = open(pdf_path + pdf_file, 'rb')
    for page in PDFPage.get_pages(of_pdf, caching=True, check_extractable=True):
        interpreter_convert.process_page(page) 
        interpreter_fetch.process_page(page)
        layout = device.get_result()
        dd = deque(obj for obj in layout if isinstance(obj,LTTextBox))
        if dd:  
            headers.append(dd.pop().get_text().replace("\n", " "))
            headers.append(dd.popleft().get_text().replace("\n", " "))
     
    text = removeNotAscii(file_handler.getvalue())
    text = re.sub(r"(\x0C|\xF0B7)"," ",text)
    text = re.sub(r"(?i)^.+?(abstrak|abstract)", "Abstrak", text, 1)
    text = re.sub(r"(?i)\d{0,2}\.*\s*daftar pustaka.*", ".", text)
    text = re.sub(r"\d{0,2}\.*\s*REFERENSI.*", ".", text)
    text = re.sub(r"(?i)\.+\s\d{0,2}\.*\s*(referensi|bibliography|references).*", ".", text)

    converter.close()
    file_handler.close()
    of_pdf.close()

#removing headers and footers
    temp = list(word[5:-5] for word in headers)
    exceptions = ["jurnal", "e-jurnal", "issn", "e-issn", "cybermatika", "ijccs"]
    pageno_pre = r"^(\s*\S{0,3}\d{2,}\s+)"
    pageno_post = r"(\s+[A-Z]*[-]*\d{2,3}\s*$)"

    for sentence in headers:
        if not(sentence.islower() or sentence.isnumeric() or len(sentence)<6 or 
        len(sentence)>130) and (temp.count(sentence[5:-5]) > 1 or 
        any(word in sentence.strip().lower() for word in exceptions)):
            sentence = removeNotAscii(sentence).rstrip()
            if re.search(pageno_pre, sentence):
                sentence = re.sub(pageno_pre, "", sentence)
            elif re.search(pageno_post, sentence):
                sentence = re.sub(pageno_post, "", sentence)
            if(sentence != ""):
                text = text.replace(sentence, "doSumnTitle" )

#final cleaning helper and finishing(writing to txt file)
    sentences = sent_tokenize(text)
    text = "\n".join(sentences)
    text = re.sub(r"((\s*[A-Z]*[-]*\d{2,}\s+doSumnTitle | doSumnTitle\s+[A-Z]*[-]*\d{2,}\s+)|doSumnTitle)"," ",text)
    text = re.sub(r"(\d)(\.)([A-Z])", r"\1. \3", text)
    text = re.sub(r"(\d+\.\s[A-Z ]+)(\d\.)", r"\1""\n"r"\2", text)
    text = re.sub("\n"r"((\d\.)+)""\n", "\n"r"\1", text)
    text = re.sub(r"(\d)(\.)([A-Z])", r"\1. \3", text)
    text = re.sub(r"(Gambar\s*\d\.)""\n",r"\1 ",text)
    text = text.replace('dkk.\n,','dkk.,')

    out = open(txt_path + txt_file,"w+")
    out.write(text)
    out.close()

if __name__ == "__main__":
    for pdf_file in os.listdir(pdf_path):
        txt_file = pdf_file.replace(".pdf",".txt")

        if txt_file in os.listdir(txt_path):
            print(txt_file+" already exists")
        else:
            print("Converting" + " " + pdf_file + " to .txt  ...")
            prepare_data(pdf_file, txt_file)