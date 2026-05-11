import csv
import re
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from helpers import _questions, get_model, load_grade_prediction_model, concatanate_feedback, normalize_text
from mock_test import _answers

### Done - All steps defined
def evaluate_answer(student_answer, question_id: int):
    #Begin validating parameters
    if not student_answer or not question_id:
        raise ValueError("Resposta do aluno e ID da questão são obrigatórias")
    
    #Get question data to compare
    reference_answer = []
    keywords = []
    for q in _questions.get("questions", []):
        if q["id"] == question_id:
            reference_answer = q["reference_answer"]
            keywords = q["keywords"]

    clean_reference_answers = [normalize_text(ans) for ans in reference_answer]
    clean_student_answer = normalize_text(student_answer)

    #If question not found or without reference answer/keywords, raise error
    if not reference_answer or not keywords:
        raise ValueError(f"Questão com ID {question_id} não encontrada ou sem resposta de referência/keywords")

    #Validate if student answer is identical to reference answer (ignoring case and punctuation)
    for answer in clean_reference_answers:
        if try_same_text(answer, clean_student_answer):
            print("Resposta idêntica à referência, atribuindo nota máxima...")
            return {
                "score": 10.0,
                "feedback": "Perfeito meu amigo, copiou do gabarito, ta colando né?"
            }
    
    #If user input is not the same as reference answer, validate keywords presence and calculate semantic similarity
    keywords_score, missing_keywords = validate_keywords(keywords, clean_student_answer)
    if keywords_score == 0:
        return {
            "score": 0.0,
            "feedback": "Nenhuma palavra-chave encontrada, revise os conceitos e tente novamente."
        }
    
    semantic_similarity_score, semantic_similarity_feedback = validate_semantic_similarity(clean_reference_answers, clean_student_answer)
    final_score = (0.40*keywords_score) + (0.60*semantic_similarity_score)
    formatted_score = round(final_score, 2)
    
    return {
        "score": formatted_score,
        "feedback": concatanate_feedback(missing_keywords, semantic_similarity_feedback)
    }

### Done - Check if user input is the same as reference answer (ignoring case and punctuation)
def try_same_text(reference_answer, student_answer):
    palavras1 = re.findall(r'\w+', reference_answer.lower())
    palavras2 = re.findall(r'\w+', student_answer.lower())

    total = len(palavras1)
    iguais = 0

    for palavra in palavras1:
        if palavra in palavras2:
            iguais += 1

    similaridade = iguais / total
    return similaridade >= 0.90

### Done - checking if keyword is used, calculate the score for each and return all missing keywords + final keyword score 
def validate_keywords(keywords, student_answer):
    quantidade_keywords = len(keywords)
    
    if quantidade_keywords == 0:
        return 0, []

    peso_keyword = 10 / quantidade_keywords
    keyword_score = 0
    missing_keywords = []

    for keyword in keywords:
        keyword_lower = keyword.lower()
        
        if keyword_lower in student_answer.lower():
            keyword_score += peso_keyword
        else:
            missing_keywords.append(keyword)

    return keyword_score, missing_keywords

### Under development
def validate_semantic_similarity(reference_answers, student_answer):
    # Carrega o modelo semântico
    model = get_model()
    similarity = []

    for reference_answer in reference_answers:
        # Gera embeddings para ambas as respostas
        embeddings = model.encode([student_answer, reference_answer])
        
        # Calcula a similaridade semântica
        similarity.append(float(cosine_similarity(
            [embeddings[0]],
            [embeddings[1]]
        )[0][0]))
    
    # similarity_score = max(similarity) * 10 

    # Usa modelo treinado para prever a grade
    score_final = predict_grade(student_answer, reference_answer, max(similarity))
    return score_final, "feedback do que ficou faltando para melhorar a nota"

### Under development
def predict_grade(student_answer, base_answer, similarity):
    grade_predictor, grade_scaler = load_grade_prediction_model()
    
    if grade_predictor is None or grade_scaler is None:
        raise RuntimeError("Modelo de predição de grades não encontrado. Execute o treinamento e salve o modelo antes de avaliar.")
    
    try:
        features = extract_features(student_answer, base_answer, similarity)
        features_scaled = grade_scaler.transform(features)
        predicted_grade = grade_predictor.predict(features_scaled)[0]
        return np.clip(float(predicted_grade), 0.0, 10.0)
    except Exception as e:
        print(f"⚠️ Erro ao prever grade: {e}")
        raise

### Under development
def extract_features(student_answer, base_answer, similarity):

    student_words = len(student_answer.split()) if student_answer else 0
    base_words = len(base_answer.split()) if base_answer else 0
    # Contar sentenças com pontuação básica
    student_sentences = max(1, len([s for s in re.split(r"[\.\!?]+", student_answer) if s.strip()]))
    
    length_ratio = student_words / base_words if base_words > 0 else 0
    
    return np.array([[
        similarity,
        length_ratio,
        student_words,
        student_sentences,
    ]])

# def test_evaluate_answer():
#     csv_path = 'api/results.csv'
#     with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.writer(csvfile)
    
#         # Cabeçalho do CSV
#         writer.writerow(['question_id', 'answer', 'original_grade', 'evaluated_grade'])
            
#         for id in _answers.get('answers', []):
#             respostas = id.get('answers')

#             for resposta in respostas:
#                 answer = resposta.get('answer')
#                 resultado = evaluate_answer(answer, id.get('id'))
#                 grade = resposta.get('grade')
#                 writer.writerow([id.get('id'), answer, grade, resultado.get('score')])       

#     print("CSV gerado com sucesso!")


# test_evaluate_answer()