import numpy as np
import torch
from dreifus.image import Img
from insightface.app import FaceAnalysis
from torch import cosine_similarity


class CSIMEvaluator:

    def __init__(self):
        self._app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
        self._app.prepare(ctx_id=0, det_size=(512, 512))

    def __call__(self, predictions: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
        embeddings_prediction = []
        embeddings_reference = []
        for prediction, reference, in zip(predictions, references):
            prediction = Img.from_torch(prediction).to_numpy().img
            reference = Img.from_torch(reference).to_numpy().img

            faces_prediction = self._app.get(prediction)
            faces_reference = self._app.get(reference)

            if len(faces_prediction) > 0 and len(faces_reference) > 0:
                embedding_prediction = faces_prediction[0]['embedding']
                embedding_reference = faces_reference[0]['embedding']

                embeddings_prediction.append(embedding_prediction)
                embeddings_reference.append(embedding_reference)

        if len(embeddings_prediction) > 0:
            embeddings_prediction = np.stack(embeddings_prediction)
            embeddings_reference = np.stack(embeddings_reference)
            csim = cosine_similarity(torch.from_numpy(embeddings_prediction), torch.from_numpy(embeddings_reference)).mean()
        else:
            csim = torch.tensor(-1)

        return csim

class UnpairedCSIMEvaluator:

    def __init__(self):
        self._app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
        self._app.prepare(ctx_id=0, det_size=(512, 512))

    def __call__(self, predictions: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
        embeddings_prediction = []
        embeddings_reference = []
        for prediction in predictions:
            prediction = Img.from_torch(prediction).to_numpy().img

            faces_prediction = self._app.get(prediction)

            if len(faces_prediction):
                embedding_prediction = faces_prediction[0]['embedding']

                embeddings_prediction.append(embedding_prediction)

        for reference in references:
            reference = Img.from_torch(reference).to_numpy().img

            faces_reference = self._app.get(reference)

            if len(faces_reference) > 0:
                embedding_reference = faces_reference[0]['embedding']

                embeddings_reference.append(embedding_reference)

        embedding_prediction = np.stack(embeddings_prediction).mean(axis=0)
        embedding_reference = np.stack(embeddings_reference).mean(axis=0)
        csim = cosine_similarity(torch.from_numpy(embedding_prediction)[None], torch.from_numpy(embedding_reference)[None])

        return csim
