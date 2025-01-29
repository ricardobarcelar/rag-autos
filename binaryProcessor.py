import fitz  # PyMuPDF
import pytesseract
import ffmpeg
import os
import json
import logging

# Configuração do logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFProcessor:
    def processar_pdf(self, caminho):
        try:
            logger.info("Processando PDF: %s", caminho)
            texto = ""
            with fitz.open(caminho) as pdf:
                for pagina in pdf:
                    texto += pagina.get_text()
            
            if not texto.strip():
                logger.info("PDF sem texto. Aplicando OCR em imagens...")
                texto = self._extrair_texto_com_ocr(pdf)
            return texto
        except Exception as e:
            logger.error("Erro ao processar PDF %s: %s", caminho, e)
            return ""
        finally:
            if os.path.exists(caminho):
                os.remove(caminho)
                logger.info("Arquivo temporário removido: %s", caminho)

    def _extrair_texto_com_ocr(self, pdf):
        texto = ""
        try:
            for pagina in pdf:
                pixmap = pagina.get_pixmap()
                imagem = pixmap.tobytes("png")
                texto += pytesseract.image_to_string(imagem, lang="por")
            return texto
        except Exception as e:
            logger.error("Erro ao aplicar OCR no PDF: %s", e)
            return ""

class ImageProcessor:
    def processar_imagem(self, caminho):
        try:
            logger.info("Processando imagem: %s", caminho)
            texto = pytesseract.image_to_string(caminho, lang="por")
            return texto
        except Exception as e:
            logger.error("Erro ao processar imagem %s: %s", caminho, e)
            return ""
        finally:
            if os.path.exists(caminho):
                os.remove(caminho)
                logger.info("Arquivo temporário removido: %s", caminho)

    @staticmethod
    def suporta_formatos(caminho):
        formatos_suportados = [".png", ".jpg", ".jpeg"]
        return any(caminho.endswith(ext) for ext in formatos_suportados)

class AudioProcessor:
    def __init__(self):
        try:
            #todo: verificar como implementar a descrição a partir do wisper
            logger.info("Carregar API Wisper")
        except Exception as e:
            logger.error("Erro ao carregar API Whisper: %s", e)
            raise

    def processar_audio(self, caminho):
        try:
            return "Policial 1: Bom dia, João. Estamos aqui para entender melhor o que aconteceu ontem. Você pode nos contar sua versão dos fatos? João: Olha, eu sei que errei, mas foi um momento de desespero. Minha filha está doente, e eu não sabia o que fazer. Policial 2: Entendemos que você estava em uma situação difícil, mas roubar não é a solução. Pode descrever como aconteceu? João: Eu estava andando pelo centro e vi uma senhora saindo de uma loja com uma bolsa cheia. Não pensei muito. Peguei a bolsa e saí correndo. Policial 1: E depois? O que fez com os itens da bolsa? João: Só tirei o dinheiro. Foram 200 reais. Depois joguei a bolsa num beco. Policial 2: Já devolvemos os pertences da vítima, mas a senhora ficou bastante abalada. Você entende a gravidade disso? João: Sim, eu entendo. Me arrependo muito. Nunca fiz nada assim antes. Policial 1: O arrependimento é importante, mas agora você terá que responder pelo que fez. Vamos encaminhar o caso para o delegado e garantir que você tenha o suporte necessário para sua filha. João: Obrigado, eu aceito as consequências. Só quero poder corrigir isso."
        except Exception as e:
            logger.error("Erro ao processar áudio %s: %s", caminho, e)
            return ""
        finally:
            if os.path.exists(caminho):
                os.remove(caminho)
                logger.info("Arquivo temporário removido: %s", caminho)

    @staticmethod
    def suporta_formatos(caminho):
        formatos_suportados = [".mp3", ".ogg", ".wav"]
        return any(caminho.endswith(ext) for ext in formatos_suportados)

class VideoProcessor:
    def __init__(self):
        self.audio_processor = AudioProcessor()

    def processar_video(self, caminho):
        audio_caminho = None
        try:
            logger.info("Processando vídeo: %s", caminho)
            audio_caminho = self._extrair_audio(caminho)
            texto = self.audio_processor.processar_audio(audio_caminho)
            return texto
        except Exception as e:
            logger.error("Erro ao processar vídeo %s: %s", caminho, e)
            return ""
        finally:
            if audio_caminho and os.path.exists(audio_caminho):
                os.remove(audio_caminho)
                logger.info("Arquivo temporário removido: %s", audio_caminho)
            if os.path.exists(caminho):
                os.remove(caminho)
                logger.info("Arquivo temporário removido: %s", caminho)

    def _extrair_audio(self, video_caminho):
        try:
            audio_caminho = f"{video_caminho}.wav"
            logger.info("Extraindo áudio do vídeo: %s", video_caminho)
            ffmpeg.input(video_caminho).output(audio_caminho, format="wav", ac=1, ar="16000").run(overwrite_output=True)
            return audio_caminho
        except Exception as e:
            logger.error("Erro ao extrair áudio do vídeo %s: %s", video_caminho, e)
            return ""

    @staticmethod
    def suporta_formatos(caminho):
        formatos_suportados = [".mp4", ".webm", ".avi"]
        return any(caminho.endswith(ext) for ext in formatos_suportados)

class FormatSupport:
    @staticmethod
    def verificar_formato(caminho):
        if caminho.endswith(".pdf"):
            return "pdf"
        elif ImageProcessor.suporta_formatos(caminho):
            return "imagem"
        elif AudioProcessor.suporta_formatos(caminho):
            return "audio"
        elif VideoProcessor.suporta_formatos(caminho):
            return "video"
        else:
            return None
