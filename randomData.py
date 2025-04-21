import pandas as pd
import numpy as np

numSamples = 250
np.random.seed(42)

experienceYears = np.random.randint(0, 10, numSamples)
technicalScore = np.random.randint(0, 101, numSamples)

hiring = np.where((experienceYears < 2) | (technicalScore < 60), 1, 0)

dataFrame = pd.DataFrame({
    'experienceYears': experienceYears,
    'technicalScore': technicalScore,
    'hiring': hiring
})

dataFrame.to_json('applicants.json', orient='records', indent=4)