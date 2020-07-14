from cleverhans.mnist_blackbox import *
from cleverhans.mnist_tutorial_jsma import *



if __name__ == "__main__":
    #run_jsma("WeightUploading/final_model.h5")
    run_black_box("WeightUploading/final_model.h5", FGSM=False)