import spacy
from typing import List
import logging

class ContentProcessor:
    def __init__(self):
        # Inicializa o spaCy com o modelo de português
        try:
            self.nlp = spacy.load("pt_core_news_sm")
            logger.info("Modelo spaCy carregado com sucesso.")
        except Exception as e:
            logger.error("Erro ao carregar o modelo spaCy: %s", e)
            raise

    def dividir_por_frases(self, texto, max_tokens=500):
        """
        Divide o texto em blocos com base em frases e limite de tokens.

        :param texto: Texto completo a ser dividido.
        :param max_tokens: Número máximo de tokens por bloco.
        :return: Lista de blocos de texto.
        """
        try:
            doc = self.nlp(texto)  # Processar o texto
            frases = [sent.text for sent in doc.sents]  # Extrair frases
            blocos = []
            bloco_atual = []

            token_count = 0
            for frase in frases:
                num_tokens = len(frase.split())  # Contar palavras como proxy para tokens
                if token_count + num_tokens > max_tokens:
                    # Salvar o bloco atual e reiniciar
                    blocos.append(" ".join(bloco_atual))
                    bloco_atual = []
                    token_count = 0

                # Adicionar a frase ao bloco atual
                bloco_atual.append(frase)
                token_count += num_tokens

            # Adicionar o último bloco
            if bloco_atual:
                blocos.append(" ".join(bloco_atual))

            return blocos
        except Exception as e:
            logger.error("Erro ao dividir texto em frases: %s", e)
            return []