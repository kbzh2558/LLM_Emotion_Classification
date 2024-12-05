# 🧠📊 Emotion Classification with Traditional Models and Transformers  
![](https://img.shields.io/badge/python-3.10%2B-blue?logo=Python)  
![HuggingFace](https://img.shields.io/badge/huggingface-transformers-yellow?logo=huggingface)  
![RandomForest](https://img.shields.io/badge/Random%20Forest-parametric_model-green)
![NaiveBayes](https://img.shields.io/badge/Naive%20Bayes-nonparametric_model-lightblue)  
![BERT](https://img.shields.io/badge/BERT-fine_tuning-red)  
![GBT-2](https://img.shields.io/badge/GBT2-fine_tuning-lightgray)  

🔍 A Comparative Study of Random Forest, Naive Bayes, and Transformer-Based Models for Emotion Classification 🔍  

> **Authors**  
> Mingshu Liu, Kaibo Zhang, and Alek Bedard  
> **Affiliation**: McGill University, This project is carried out under the supervision of Professors [Isabeau Prémont-Schwarz](https://www.cs.mcgill.ca/~isabeau/) and [Reihaneh Rabbany](http://www.reirab.com/). It is a part of the coursework of COMP551 Applied Machine Learning.

---

## Overview  

This project evaluates traditional machine learning models and transformer-based architectures for emotion classification on the **GoEmotions dataset**. It explores how model architecture, data preprocessing, and hyperparameter tuning impact performance, particularly in handling rare emotions and imbalanced data.  

**Key contributions include:**  
- Analysis of traditional models (Random Forest and Naive Bayes).  
- Evaluation of pre-trained and fine-tuned BERT and GPT-2 models.  
- Investigation of attention mechanisms in transformer models.  
- Recommendations for future improvements using advanced sampling and hybrid architectures.  

---

## Dataset Description  

The **GoEmotions dataset** consists of 58,000 Reddit comments labeled into 27 emotion categories plus a neutral class.  

- **Training Samples**: 40,000  
- **Validation Samples**: 10,000  
- **Test Samples**: 8,000  

Class imbalance is a notable challenge, with the majority class ("neutral") dominating the dataset. Minority classes like "grief" and "pride" require specialized techniques to improve model performance.  

---

### Step-by-Step Experiments  

1. <details>  
    <summary>Random Forest Baseline</summary>  

    - Leveraged bag-of-words representation for text features.  
    - Achieved **training accuracy: 99.61%** and **test accuracy: 54.12%**, indicating significant overfitting.  
    - Struggled with rare emotions due to shallow feature representations.  
   </details>  

2. <details>  
    <summary>Naive Bayes Model</summary>  

    - Tuned smoothing hyperparameter (`alpha`) for optimal performance.  
    - Test accuracy: **44.49%**, F1 score: **36.89%**, and AUC: **0.8199%.**  
    - Highlighted limitations of the independence assumption in nuanced text classification.  
   </details>  

3. <details>  
    <summary>BERT Pre-training and Fine-tuning</summary>  

    - Pre-trained BERT struggled with **test accuracy: 3.33%**.  
    - Fine-tuned BERT achieved **accuracy: 63.03%, F1 score: 61.33%, and AUC: 0.9390%.**  
    - Attention analysis revealed strengths in token-level embedding and contextual relationships but struggled with rare emotions.  
   </details>  

4. <details>  
    <summary>GPT-2 Evaluation</summary>  

    - Fine-tuned GPT-2 achieved **accuracy: 59.59%** and **AUC: 0.9186%.**  
    - Demonstrated improvements over pre-trained performance but exhibited signs of overfitting.  
   </details>  

---

## Results Summary  

| Model                | Test Accuracy | F1 Score | AUC   |  
|----------------------|---------------|----------|-------|  
| Random Forest        | 54.12%        | 46.78%   | 0.8426 |  
| Naive Bayes          | 44.49%        | 36.89%   | 0.8199 |  
| Fine-tuned BERT      | 63.03%        | 61.33%   | 0.9390 |  
| Fine-tuned GPT-2     | 59.59%        | 58.29%   | 0.9186 |  

---

## Insights and Future Work  

- **Class Imbalance:** Imbalanced class distribution significantly impacts rare emotion detection. Techniques like data augmentation are essential for improvement.  
- **BERT's Attention Mechanism:** Offers superior performance in capturing semantic relationships but struggles with ambiguous or polarizing sentences.  
- **GPT-2 Performance:** Showcases potential in nuanced emotion detection, albeit with limitations in overfitting.  

**Future Directions:**  
- Incorporate hybrid architectures (e.g., CNN-Transformer).  
- Experiment with advanced sampling techniques.  
- Utilize interpretability tools like Grad-CAM to enhance model insights.  

---

## Citation  
[1] Dorottya Demszky, Dana Movshovitz-Attias, Jeongwoo Ko, Alan Cowen, Gaurav Nemade, and Sujith Ravi.
GoEmotions: A Dataset of Fine-Grained Emotions. In 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020.
[2] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional
transformers for language understanding. CoRR, abs/1810.04805, 2018.
[3] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are
unsupervised multitask learners. 2019.
