from prepare_model import Classifier
from paths import path2params
cls = Classifier()
cls.train(show_progress=True)
cls.load(path2params)
cls.parse('転職したい')
print()
