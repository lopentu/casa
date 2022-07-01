# CASA

## Scripts

### 把中華電的原始資料轉成OpinionThreads

```bash
python ../bin/proc/make_threads.py raw-data-cht-202010.csv raw-data-cht-202010.pkl
```

## 套件說明

`cadence`：主要程式介面，該程式介面負責協調控制三項次模組：`cadet`、`crytal`、`MTBert`
- `cadet` (Cht Absa DETection)：負責以辭典和詞向量偵測業者和服務面向
- `Crytal` ：負責以服務知識本體和構式偵測句子中的評論對象和評價極度。
- `MTBert` (Multi-Task BERT)：負責呼叫並整理深度學習模型 (i.e. BERT) 所產生之預測內容。

上述成分整理為以下主要目錄：
- `bin`: 執行腳本 (script)。此腳本為Q2交付內容，其功能為重新建立服務面向偵測或進行新詞探勘等作業。若僅需使用casa，則無須額外執行此目錄之腳本內容。
- `data`: 所有與 `casa` 相關之資料。這些資料安排為以下資料夾：
  - `annot-data`：原始標記資料。其中 annot_frame_90(91-99).csv 為標記者使用 LabelStudio 對資料標記之原始檔案。從這些原始檔案中，再自動整理成 merged_frame_20210722.csv 以及 aspect_tuples_20210722.csv。
  - `cadence`: 初始化 `cadence` 物件所需之 config.json。該資料夾中其他檔案為 Q3 交付之舊版展示用模型和參數檔。
  - `cadet`：業者與服務面向偵測用參數檔。該資料夾中有三個版本之參數，最新版本為 op20.3。每個版本裡的參數檔案結構均相同，這些檔案包括 ft-2020.kv (詞向量檔案) ，seeds.csv (服務面向辭典) ，spm-2020.(model|vocab)  (新詞探勘之 sentencepiece 模型) 。
  - `casper`：Q3 所展示之統計報表所用之原始資料。
  - `crystal`：該資料夾包含 constructions.csv (構式辭典) ，以及 eval_ontology.csv (服務面向架構) 。
  - `font`：Q2 交付程式中所使用的字形檔
  - `mtbert`：MTBert的模型檔案，該模型權重為 pyTorch 之儲存格式。該模型權重對應之程式檔請參見原始碼資料夾下之 mtbert.py。
- `etc`：此資料夾中的檔案為 `casa` 程式使用範例。所有使用範例皆以 Jupyter notebook 方式呈現。其中 90.31-Q4-cadence.ipynb 是最新的 `cadence` 執行範例，其他檔案僅為 Q2、Q3報告中所使用之展示檔案。
- `src`：主要程式使用介面。該資料夾下之 `casa` 為一 Python 模組。該模組下之資料夾結構直接對應 `cadence` 的次模組結構：`cadence` 為程式介面、`cadet` 為服務面向偵測、`crystal` 為服務面向架構、`MTBert` 為深度學習模型。
