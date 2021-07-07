import logging
import re
from tqdm.auto import tqdm

def default_field_mapper(data_dict):
    field_map = {
        '輿情ID': 'id',       '輿情標題': 'title',
        '輿情來源': 'source',  '輿情頻道': 'channel',
        '輿情網站': 'source',
        '輿情作者': 'author',  '輿情內文': 'text',
        '主文/回文': 'post_type', '原始網址': 'url',
        '議題類別': 'topic',
        "(修改後)燈號": "sentence_sentiment"
    }
    return {field_map[k]: v for k, v in data_dict.items() if k in field_map}

url_pat = re.compile(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b"
                        "([-a-zA-Z0-9()@:%_\+.~#?&//=]*)")
def remove_url(x):
    return url_pat.sub("", x)
class Opinion:
    def __init__(self, data_dict, field_mapper=default_field_mapper):        
        data_dict = field_mapper(data_dict)
        self.id = data_dict.get("id", "").strip()
        self.title = remove_url(data_dict.get("title", "")).strip()
        self.source = data_dict.get("source", "").strip()
        self.channel = data_dict.get("channel", "").strip()
        self.author = data_dict.get("author", "").strip()
        self.text = remove_url(data_dict.get("text", "")).strip()
        self.post_type = data_dict.get("post_type", "").strip()
        self.url = data_dict.get("url", "").strip()
        self.topic = data_dict.get("topic", "")
        self.sentence_sentiment = data_dict.get("sentence_sentiment", None)     
    
    def __repr__(self):
        text = self.text[:20]+"..." if len(self.text)>20 else self.text
        return f"<Opinion [{self.source}] {self.author}: {text}>"
    
    def merge_previous(self, prev_opinion):
        assert self.id != prev_opinion.author
        assert self.author == prev_opinion.author
        assert self.title == prev_opinion.title
        if self.sentence_sentiment != prev_opinion.sentence_sentiment:
            self.sentence_sentiment = None

        self.text = prev_opinion.text + " " + self.text        





    