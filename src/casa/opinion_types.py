import logging
from tqdm.auto import tqdm

MAIN_LABELS = ["主文"]

def default_field_mapper(data_dict):
    field_map = {
        '輿情ID': 'id',       '輿情標題': 'title',
        '輿情來源': 'source',  '輿情頻道': 'channel',
        '輿情作者': 'author',  '輿情內文': 'text',
        '主文/回文': 'post_type', '原始網址': 'url',
        '議題類別': 'topic'
    }
    return {field_map[k]: v for k, v in data_dict.items() if k in field_map}

class Opinion:
    def __init__(self, data_dict, field_mapper=default_field_mapper):
        data_dict = field_mapper(data_dict)
        self.id = data_dict.get("id", "").strip()
        self.title = data_dict.get("title", "").strip()
        self.source = data_dict.get("source", "").strip()
        self.channel = data_dict.get("channel", "").strip()
        self.author = data_dict.get("author", "").strip()
        self.text = data_dict.get("text", "").strip()
        self.post_type = data_dict.get("post_type", "").strip()
        self.url = data_dict.get("url", "").strip()
        self.topic = data_dict.get("topic", "")
        self.sentence_sentiment = None        
    
    def __repr__(self):
        text = self.text[:20]+"..." if len(self.text)>20 else self.text
        return f"<Opinion [{self.source}] {self.author}: {text}>"
    
    def merge_previous(self, prev_opinion):
        assert self.id != prev_opinion.author
        assert self.author == prev_opinion.author
        assert self.title == prev_opinion.title
        self.text = prev_opinion.text + " " + self.text        

class OpinionThread:
    def __init__(self):
        self.main = None
        self.replies = []

    def get_opinion(self):
        if self.main:
            return self.main
        elif self.replies:
            return self.replies[0]
        else:
            return None

    @property
    def title(self):
        op = self.get_opinion()
        return op.title if op else ""
    
    @property
    def channel(self):
        op = self.get_opinion()
        return op.channel if op else ""

    @property
    def source(self):
        op = self.get_opinion()
        return op.source if op else ""

    def __len__(self):
        return bool(self.main) + len(self.replies)

    def add(self, opinion: Opinion):
        if self.main and opinion.post_type == "main":
            logging.warning("Main already set, add to replies")
            self.replies.append(opinion)
            return
        
        if opinion.post_type in MAIN_LABELS:
            self.main = opinion
        else:
            if opinion.id in [x.id for x in self.replies]:
                pass
            self.replies.append(opinion)
    
    def __repr__(self):        
        if self.main:
            author = self.main.author
            title = self.main.title
            return f"<OpinionThread[T]: {title}/{author}/{len(self.replies)} replie(s)>"
        else:            
            title = self.replies[0].title if self.replies else "NA"
            author = self.replies[0].author if self.replies else "NA"
            return f"<OpinionThread[R]: {title}/{author}/{len(self.replies)} replie(s)>"
    
    def print_thread(self):
        if self.main:
            print("Main:", self.main)
        else:
            print("Main: (No main post)")
        for reply in self.replies:
            print("--", reply)
    
    def opinion_texts(self):
        if self.main:
            yield self.main.title + "\u3000" + self.main.text
            for reply_x in self.replies:
                yield reply_x.text
        else:
            if not self.replies: return
            reply0 = self.replies[0]
            yield reply0.title + "\u3000" + reply0.text
            
            if len(self.replies) > 1:
                for reply_x in self.replies[1:]:
                    yield reply_x.text
        

def make_opinion_threads(data):
    last_opinion = Opinion({})    
    thread_list = []
    thread = OpinionThread()    
    for data_item in tqdm(data):        
        try:
            opinion = Opinion(data_item)
        except Exception as ex:
            print(ex)
            continue

        if last_opinion.title == opinion.title:            
            # only merge opinion when opinion source is PTT
            if last_opinion.source.lower() == "ptt" and \
                (last_opinion.author == opinion.author):
                opinion.merge_previous(last_opinion)                                                    
            else:
                thread.add(last_opinion)                                

        else:
            if last_opinion.id:
                thread.add(last_opinion)
            if thread.main or thread.replies:
                thread_list.append(thread)

            thread = OpinionThread()
        last_opinion = opinion

    if last_opinion.id:
        thread.add(last_opinion)
    if thread.main or thread.replies:    
        thread_list.append(thread)
    
    return thread_list


    