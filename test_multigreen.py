import torch
from src.masks.multigreen import MaskCollator

if __name__ == "__main__":
    collator = MaskCollator(
        input_size=(224, 224),
        patch_size=16,
        enc_mask_scale=(0.3, 0.4), 
        pred_mask_scale=(0.1, 0.2), 
        nenc=1,
        npred=2,
        min_keep=10,
        allow_overlap=False,
        data_path="./dev/colorMAE/green_noise_data_3072.npz",
    )
    
    # Create dummy batch of 4 images
    dummy_batch = [torch.randn(3, 224, 224) for _ in range(4)]
    
    # Run collation
    collated_batch, collated_masks_enc, collated_masks_pred = collator(dummy_batch)
    
    print("Collated Batch Shape:", collated_batch.shape)
    
    print("\nEncoder Masks List Length:", len(collated_masks_enc))
    for i, enc_tensor in enumerate(collated_masks_enc):
        print(f"  Encoder Mask {i} Type/Shape:", type(enc_tensor), enc_tensor.shape)
        
    print("\nPredictor Masks List Length:", len(collated_masks_pred))
    for i, pred_tensor in enumerate(collated_masks_pred):
        print(f"  Predictor Mask {i} Type/Shape:", type(pred_tensor), pred_tensor.shape)

    # Check for overlaps
    print("\nChecking for overlaps...")
    bidx = 0  # check first image batch
    enc_mask = collated_masks_enc[0][bidx]  # shape: [min_keep_enc]
    pred_mask_1 = collated_masks_pred[0][bidx]  # shape: [min_keep_pred]
    pred_mask_2 = collated_masks_pred[1][bidx]  # shape: [min_keep_pred]
    
    def check_overlap(t1, t2, name1, name2):
        s1 = set(t1.tolist())
        s2 = set(t2.tolist())
        intersection = s1.intersection(s2)
        if intersection:
            print(f"  [ERROR] Overlap between {name1} and {name2}! Indices: {intersection}")
        else:
            print(f"  [OK] No overlap between {name1} and {name2}")
            
    check_overlap(enc_mask, pred_mask_1, "E0", "P0")
    check_overlap(enc_mask, pred_mask_2, "E0", "P1")
