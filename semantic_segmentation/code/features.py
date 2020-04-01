from helper import *


def get_features_trad(student, teacher):
    sf_student = [SaveFeatures(m) for m in [student.encoder.layer2,
                                            student.decoder.blocks[2],
                                            ]]

    sf_teacher = [SaveFeatures(m) for m in [teacher.encoder.layer2,
                                            teacher.decoder.blocks[2],
                                            ]]
    sf_student, sf_teacher


def get_features(student, teacher):
    sf_student = [SaveFeatures(m) for m in [student.encoder.relu,
                                            student.encoder.layer1,
                                            student.encoder.layer2,
                                            student.encoder.layer3,
                                            student.encoder.layer4,
                                            student.decoder.blocks[0],
                                            student.decoder.blocks[1],
                                            student.decoder.blocks[2],
                                            student.decoder.blocks[3],
                                            student.decoder.blocks[4]
                                            ]]

    sf_teacher = [SaveFeatures(m) for m in [teacher.encoder.relu,
                                            teacher.encoder.layer1,
                                            teacher.encoder.layer2,
                                            teacher.encoder.layer3,
                                            teacher.encoder.layer4,
                                            teacher.decoder.blocks[0],
                                            teacher.decoder.blocks[1],
                                            teacher.decoder.blocks[2],
                                            teacher.decoder.blocks[3],
                                            teacher.decoder.blocks[4]
                                            ]]
    return sf_student, sf_teacher
