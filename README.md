# Key Information Extraction From Documents: Evaluation And Generator

Natural language processing methods are usually used on one dimensional sequences of text. In some cases, for example, for the extraction of key information of invoice- or order-documents, semantic information such as the position of text or font-sizes are crucial to understand the contextual meaning. Natural language processing methods show deficits regarding this task since the semantic information is not processed. Convolutional neural networks are already common in computer vision models to process and extract relationships in multidimensional data. Therefore, natural language processing models have already been combined with computer vision models in the past, to benefit from e.g. positional information and to improve performance of these key information extraction models. Existing models were either trained on not published data sets or on an annotated collection of receipts, which did not focus on PDF-like documents. Hence, in this research project a template-based document generator was created to compare SoA models for information extraction. An existing information extraction model “Chargrid” (Katti et al., 2019) was reconstructed and the impact of a bounding box regression decoder, as well as the impact of an NLP pre-processing step was evaluated for information extraction from documents. The results have shown that an NLP based pre-processing step is beneficial for model performance. However, the use of a bounding box regression decoder increases the model performance only for fields that do not follow a rectangle shape.

## src

src/ contains the source files for the project.
Settings can be adjusted with the settings.ini file.

1_0_Company_Generator:  
Generation of fake companies

1_1_PDF_Generator:
Generate PDF documents based on meta information and templates.

2_Ground_Truth_Generator:
Generate ground truth data for model training based on documents and meta information.

3_1_CharGrid_PreProcessing and 4_1_SpacyGrid_PreProcessing:
Pre-processing step for Chargrid / SpaCygrid data.

3_2_CharGrid_Model and 4_2_SpacyGrid_Model:
Chargrid / SpaCygrid model.

3_3_CharGrid_Eval and 4_3_SpacyGrid_Eval:
Predict documents with a trained chargrid model.

5_1_Evaluation:
Evaluation of the predicted documents compared to ground truth data.

## data

The data folder can be found here:
https://drive.google.com/drive/folders/1ZSCW_LthTHUpigIDgfijLOR5ZSbksg_g
