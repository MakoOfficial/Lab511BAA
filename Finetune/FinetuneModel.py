from ContrastLearning.contrast_model import get_student_contrast_model


def get_fine_model(student_path):
    backbone = get_student()
    if student_path is not None:
        backbone.load_state_dict(torch.load(student_path))

    resnet, output_channels = get_pretrained_resnet50(True)

    return Student_Contrast_Model(backbone, resnet)