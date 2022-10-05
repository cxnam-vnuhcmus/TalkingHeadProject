from imaginaire.evaluation.lpips import get_lpips_model

def calculate_LPIPS():
    model = get_lpips_model()
    metric_dict = model(fake_img)
    return metric_dict