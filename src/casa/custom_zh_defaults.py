from spacy.attrs import LANG
from spacy.language import Language
from spacy.lang.zh import Chinese, LEX_ATTRS
from spacy.lang.tokenizer_exceptions import BASE_EXCEPTIONS
from spacy.lang.zh.lex_attrs import LEX_ATTRS
from spacy.lang.zh.stop_words import STOP_WORDS
from spacy.lang.zh.tag_map import TAG_MAP
from spacy.util import DummyTokenizer

from spacy.tokens import Doc
from spacy.util import get_words_and_spaces

class IdentityTokenizer(DummyTokenizer):
    def __init__(self, cls, nlp=None, config={}):
        self.vocab = nlp.vocab if nlp is not None else cls.create_vocab(nlp)
    
    def __call__(self, presegs):         
        text = presegs["raw"]
        words = presegs["words"]
        (words, spaces) = get_words_and_spaces(words, text)
        return Doc(self.vocab, words=words, spaces=spaces)

class CustomChineseDefaults(Language.Defaults):
    lex_attr_getters = dict(Language.Defaults.lex_attr_getters)
    lex_attr_getters.update(LEX_ATTRS)
    lex_attr_getters[LANG] = lambda text: "zh"
    tokenizer_exceptions = BASE_EXCEPTIONS
    stop_words = STOP_WORDS
    tag_map = TAG_MAP
    writing_system = {"direction": "ltr", "has_case": False, "has_letters": False}
    use_jieba = True
    use_pkuseg = False

    @classmethod
    def create_tokenizer(cls, nlp=None, config={}):
        print("nlp", nlp)
        return IdentityTokenizer(cls, nlp, config=config)
