from recognition_of_digits.main import RecognitionOfDigits
import numpy as np


def main():
    rod = RecognitionOfDigits("F:\\machine_learning_workspace\\MLAction\\date_set\\RecognitionOfDigits\\train.txt",
                        "F:\\machine_learning_workspace\\MLAction\date_set\\RecognitionOfDigits\\test.txt")
    predict_y,score = rod.predict()
    print(score)
    np.set_printoptions(threshold=np.inf)
    print(predict_y)


if __name__ == '__main__':
    main()