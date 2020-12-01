import logging
from tqdm.auto import tqdm
from .opinion_types import Opinion

MAIN_LABELS = ["主文"]

class OpinionThread:
    def __init__(self):
        self.main = None
        self.replies = []
        self.proc_flags = []

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
        flag = "|".join(self.proc_flags) if self.proc_flags else ""        
        if flag:
            flag = f" ({flag})"

        if self.main:
            author = self.main.author
            title = self.main.title            
            return f"<OpinionThread[T]: {title}/{author}/{len(self.replies)} replie(s){flag}>"
        else:            
            title = self.replies[0].title if self.replies else "NA"
            author = self.replies[0].author if self.replies else "NA"
            return f"<OpinionThread[R]: {title}/{author}/{len(self.replies)} replie(s){flag}>"
    
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
    
    def opinion_tokens(self):
        def get_tokens(op):
            text_tokens = getattr(op, "text_tokens", [])
            title_tokens = getattr(op, "title_tokens", [])
            return title_tokens, text_tokens                
        
        if self.main:
            title_tokens, text_tokens = get_tokens(self.main)
            if title_tokens or text_tokens:
                yield title_tokens + text_tokens
            for reply_x in self.replies:
                title_tokens, text_tokens = get_tokens(reply_x)
                yield text_tokens
        else:
            if not self.replies: return
            reply0 = self.replies[0]
            title_tokens, text_tokens = get_tokens(reply0)
            yield title_tokens + text_tokens
            
            if len(self.replies) > 1:
                for reply_x in self.replies[1:]:
                    title_tokens, text_tokens = get_tokens(reply_x)
                    yield text_tokens

    def process(self, processor):
        if processor.flag in self.proc_flags:
            # logging.warning("Thread is already processed with %s", str(processor))
            pass
        if self.main:
            self.main = processor.process(self.main)
        
        self.replies = [processor.process(x) for x in self.replies]
        self.proc_flags.append(processor.flag)
        



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