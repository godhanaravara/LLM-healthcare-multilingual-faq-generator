## ▸ Overview

The LLM-based**Healthcare FAQ Generator** is a full-stack NLP pipeline that generates accurate, multilingual responses to medical questions, supporting telehealth applications. It processes the Medical FAQ dataset (`medquad.csv`, 16,412 rows) through data cleaning, preprocessing, fine-tuning a `google/flan-t5-base` model, Retrieval-Augmented Generation (RAG) with LangChain and FAISS, and translation into Spanish and Telugu using the GCP Translation API. The pipeline achieves a ROUGE-1 score of 0.316 and delivers 331 `Correct` answers, enhancing patient accessibility and reducing clinician workload by up to 60% (per Azure economics).

This project showcases my expertise in **data engineering** (PySpark, Azure Blob Storage), **NLP** (Hugging Face, LangChain, FAISS), and **cloud integration** (GCP, Azure).

---
## ▸ Tech Stack
This project leverages:

- ![Databricks](https://img.shields.io/badge/Databricks-EC6B24?style=for-the-badge&logo=databricks&logoColor=white) + ![Apache Spark](https://img.shields.io/badge/Apache_Spark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white): Scalable environment for distributed processing of 16,359 medical FAQs.
- ![PySpark](https://img.shields.io/badge/PySpark-FF5722?style=for-the-badge&logo=apachespark&logoColor=white): Handles data I/O, transformations, and Parquet storage for efficient pipelines.
- ![Hugging Face Transformers](https://img.shields.io/badge/Hugging%20Face_Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black): Fine-tunes `google/flan-t5-base` (250M parameters) and provides evaluation metrics (ROUGE-1: 0.316).
- ![LangChain](https://img.shields.io/badge/LangChain-0E83CD?style=for-the-badge): RAG pipeline orchestration for context-aware FAQ generation.
- ![FAISS](https://img.shields.io/badge/FAISS-005FED?style=for-the-badge): Vector indexing and similarity search.
- ![Sentence-Transformers](https://img.shields.io/badge/Sentence--Transformers-1E88E5?style=for-the-badge): Embeddings (`all-MiniLM-L6-v2`) for RAG retrieval.
- ![NLTK](https://img.shields.io/badge/NLTK-4CAF50?style=for-the-badge&logo=python&logoColor=white): Tokenization and lemmatization for preprocessing medical questions.
- ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white) + ![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white): Classic ML utilities and tabular processing.
- ![Google Cloud Translate](https://img.shields.io/badge/Google_Cloud_Translate-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white): Multilingual answer generation (Spanish, Telugu).
- ![Azure Blob Storage](https://img.shields.io/badge/Azure_Blob_Storage-0089D6?style=for-the-badge&logo=microsoftazure&logoColor=white): Dataset and artifact storage.
- ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) + ![python-dotenv](https://img.shields.io/badge/python--dotenv-607D8B?style=for-the-badge): Core language and environment management.
- ![openpyxl](https://img.shields.io/badge/openpyxl-3DDC84?style=for-the-badge) + ![tqdm](https://img.shields.io/badge/tqdm-343434?style=for-the-badge): Export and progress tracking.

---
## ▸ Features

- **Scalable Data Pipeline**: Processes 16,412 medical FAQs using PySpark, cleaning and transforming to 16,359 rows in Parquet format.
- **NLP Preprocessing**: Lemmatizes questions with NLTK and Spark UDFs, ensuring consistent text for model training.
- **LLM Fine-Tuning**: Fine-tunes `google/flan-t5-base` (250M parameters, 3 epochs) on 16,359 QA pairs, achieving eval_loss ~1.589.
- **RAG Pipeline**: Uses LangChain and FAISS to ground answers, evaluating 500 samples with TF-IDF similarity (331 `Correct`, 127 `Partially correct`, 42 `Incorrect`).
- **Multilingual Support**: Translates 331 `Correct` answers into Spanish (e.g., “Herencia dominante autosómica”) and Telugu (e.g., “ఆటోసోమల్ డామినెంట్”) using GCP Translation API.
- **Evaluation**: Computes ROUGE scores (ROUGE-1: 0.316, ROUGE-L: 0.248) and verdict assignments for quality assurance.
- **Cloud Integration**: Leverages Azure Blob Storage for data and model storage, GCP for translations, and Databricks for processing.
---
## ▸ Business Impact

- **Patient Accessibility**: Delivers multilingual FAQs (Spanish, Telugu) for diverse patient populations, improving engagement and equity.
- **Clinical Efficiency**: Automates accurate FAQ responses, reducing clinician workload by up to 60% (per Azure economics).
- **Scalability**: Supports extensible languages (e.g., French, Hindi) and larger datasets, adaptable for global telehealth platforms.
- **Cost Savings**: Uses Parquet compression and CPU-friendly processing, minimizing compute costs on Databricks Community Edition.

---
## ▸ Installation

### Prerequisites
- Python 3.12
- Databricks Community Edition (CPU)
- GCP service account key (`healthcare-faq-translation-key.json`)
- Azure Blob Storage with SAS token
- Dependencies:
  ```bash
  pip install pandas pyspark==3.5.0 nltk transformers datasets evaluate langchain sentence-transformers faiss-cpu scikit-learn tqdm google-cloud-translate==2.0.1 python-dotenv openpyxl

---
## ▸ Setup
1. Clone the repository:

```python
git clone https://github.com/godhanaravara/LLM-healthcare-multilingual-faq-generator.git
cd healthcare-faq-generator
```

2. Set up environment variables in .env:

```python
AZURE_STORAGE_ACCOUNT=<your-account>
AZURE_CONTAINER=<your-container>
AZURE_SAS_TOKEN=<your-sas-token>
MOUNT_PT=/mnt/faqdata
GOOGLE_APPLICATION_CREDENTIALS=/path/to/healthcare-faq-translation-key.json
```

3. Install dependencies:
```python
pip install -r requirements.txt
```

4. Download `medquad.csv` (16,412 rows) and place it in your Azure Blob Storage container.

---
## ▸ Usage
Run the notebooks sequentially in Databricks or WSL Jupyter:

1. `01_load_clean_data.ipynb`: Loads `medquad.csv`, cleans (lowercase, remove special characters), and saves `transformed_medquad.parquet` (16,359 rows). (~1-2 min)
```python
spark-submit 01_load_clean_data.ipynb
```
2. `02_preprocess_medquad.ipynb`: Lemmatizes questions using NLTK and Spark UDFs, checks for bias, and saves `preprocessed_medquad.parquet`. (~5-10 min)
3. `03_train_LLM.ipynb`: Fine-tunes `google/flan-t5-base` (3 epochs, eval_loss ~1.589), evaluates 500 samples (ROUGE-1: 0.316), and saves `eval_preds.csv`. (~37h training, ~1h33m evaluation)
4. `04_langchain_RAG.ipynb`: Builds RAG pipeline with LangChain and FAISS, generates answers, assigns verdicts (331 Correct), and saves `rag_faq_eval.csv`. (~1h33m)
5. `05_GCP_translate.ipynb`: Translates 331 Correct answers to Spanish and Telugu, saving `rag_faq_eval_translated.xlsx`. (~60-70s per language)

---
## ▸ Outputs
- `transformed_medquad.parquet`: Cleaned dataset (16,359 rows).
- `preprocessed_medquad.parquet`: Lemmatized dataset.
- `finetuned_llm_prototype/`: Fine-tuned flan-t5-base model.
- `eval_preds.csv`: 500 evaluated predictions with ROUGE scores.
- `rag_faq_eval.csv`: RAG-generated answers with verdicts.
- `rag_faq_eval_translated.xlsx`: 331 Correct answers in Spanish and Telugu.

---
## ▸ Project Structure
healthcare-faq-generator/
├── 01_load_clean_data.ipynb        # Data ingestion and cleaning
├── 02_preprocess_medquad.ipynb     # Text preprocessing and lemmatization
├── 03_train_LLM.ipynb              # LLM fine-tuning and evaluation
├── 04_langchain_RAG.ipynb          # RAG pipeline with LangChain and FAISS
├── 05_GCP_translate.ipynb          # Multilingual translation with GCP
├── requirements.txt                # Dependencies
├── .env                            # Environment variables
└── README.md                       # Project overview

---
## ▸ Results
- Data Processing:
    - Raw: 16,412 rows (medquad.csv).
    - Cleaned: 16,359 rows (transformed_medquad.parquet).
    - Preprocessed: 16,359 rows with lemmatized questions (preprocessed_medquad.parquet).

- Model Performance:
    - Fine-tuned flan-t5-base: Eval_loss ~1.589 after 3 epochs.
    - ROUGE Scores (500 samples): ROUGE-1: 0.316, ROUGE-2: 0.172, ROUGE-L: 0.248, ROUGE-Lsum: 0.253.
    - Verdicts: 331 Correct, 127 Partially correct - Needs Review, 42 Incorrect.

- Multilingual Support:
    - Translated 331 Correct answers to Spanish (e.g., “Herencia dominante autosómica”) and Telugu (e.g., “ఆటోసోమల్ డామినెంట్”).
    - Runtime: ~60-70s per language for 331 translations.

- Challenges:
    - Hallucinations in fine-tuned model (e.g., CKD nutrition errors) mitigated by RAG grounding.
    - Limited GPU access; used CPU-friendly processing on Databricks Community Edition.

 ---
## ▸ Current and Future Work
- AutoGen Integration: Developing multi-agent workflows with AutoGen to automate FAQ generation and validation, enhancing pipeline efficiency.
- AWS SageMaker Deployment: Deploying the RAG pipeline and fine-tuned model on AWS SageMaker for scalable inference.
- Streamlit App: Building an interactive Streamlit app to display multilingual FAQs, enabling user-friendly query testing.
- Expanded Languages: Adding French (`fr`) and Hindi (`hi`) translations using GCP Translation API.
- Verdict Refinement: Incorporating keyword-based rules (e.g., “beans and lentils” for CKD) to improve `Partially correct` verdicts.
- Performance Optimization: Exploring batch processing and PySpark for faster RAG evaluation on larger datasets.

---
## ▸ Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (git checkout -b feature/xyz).
3. Commit changes (git commit -m 'Add feature xyz').
4. Push to the branch (git push origin feature/xyz).
5. Open a pull request.

---
## ▸ About Me
I am a data enthusiast passionate about using AI to understand human trends. This project reflects my journey in blending technology with real-world impact. Connect with me on [LinkedIn](https://www.linkedin.com/in/yourprofile) or explore more at my [GitHub](https://github.com/godhanaravara)!