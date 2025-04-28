import torch
import torch.nn as nn

class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YOLOLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.lambda_obj = 1.0
        self.lambda_class = 1.0
        
        # MSE loss for coordinates and confidence
        self.mse = nn.MSELoss(reduction='sum')
        # BCE loss for class predictions
        self.bce = nn.BCELoss(reduction='sum')
        
    def forward(self, predictions, targets):
        """
        predictions: [batch_size, S, S, B*5 + C]
        targets: [batch_size, S, S, B*5 + C]
        Returns:
            total_loss: scalar tensor
            loss_components: dict containing individual loss components
        """
        batch_size = predictions.shape[0]
        
        # Initialize loss components
        total_loss = torch.zeros(1, device=predictions.device)
        coord_loss = torch.zeros(1, device=predictions.device)
        obj_loss = torch.zeros(1, device=predictions.device)
        noobj_loss = torch.zeros(1, device=predictions.device)
        class_loss = torch.zeros(1, device=predictions.device)
        
        # Calculate loss for each grid cell
        for i in range(self.S):
            for j in range(self.S):
                # Get predictions and targets for this cell
                pred_cell = predictions[:, i, j]  # [batch_size, B*5 + C]
                target_cell = targets[:, i, j]    # [batch_size, B*5 + C]
                
                # Calculate IoU for each box pair
                ious = torch.zeros((batch_size, self.B), device=predictions.device)
                for b in range(self.B):
                    pred_box = pred_cell[:, b*5:(b+1)*5]    # [batch_size, 5]
                    target_box = target_cell[:, :5]          # [batch_size, 5]
                    ious[:, b] = self._calculate_iou(pred_box, target_box)
                
                # Find best matching box for each sample
                best_box_idx = torch.argmax(ious, dim=1)  # [batch_size]
                
                # Calculate coordinate loss for best matching boxes
                for b in range(self.B):
                    mask = (best_box_idx == b) & (target_cell[:, 4] == 1)
                    num_coord_samples = mask.sum().item()
                    if num_coord_samples > 0:
                        # print(f"[DEBUG] coord_loss mask ON for {num_coord_samples} samples at grid cell ({i},{j}), box {b}")
                        if mask.any():
                            pred_box = pred_cell[mask, b*5:(b+1)*5]
                            target_box = target_cell[mask, :5]
                            cell_coord_loss = self.mse(pred_box[:, :4], target_box[:, :4])
                            coord_loss += cell_coord_loss
                            total_loss += self.lambda_coord * cell_coord_loss
                            # if num_coord_samples > 0 and i == 0 and j == 0 and b == 0:
                                # print(f"[DEBUG] pred_box[:3]: {pred_box[:3, :4].detach().cpu().numpy()}")
                                # print(f"[DEBUG] target_box[:3]: {target_box[:3, :4].detach().cpu().numpy()}")
                            # print(f"[DEBUG] cell_coord_loss: {cell_coord_loss.item()}")
                
                # Calculate object loss
                obj_mask = target_cell[:, 4] == 1
                if obj_mask.any():
                    for b in range(self.B):
                        pred_conf = pred_cell[obj_mask, b*5+4]
                        target_conf = target_cell[obj_mask, 4]
                        cell_obj_loss = self.mse(pred_conf, target_conf)
                        obj_loss += cell_obj_loss
                        total_loss += self.lambda_obj * cell_obj_loss
                
                # Calculate no object loss
                noobj_mask = target_cell[:, 4] == 0
                if noobj_mask.any():
                    for b in range(self.B):
                        pred_conf = pred_cell[noobj_mask, b*5+4]
                        target_conf = target_cell[noobj_mask, 4]
                        cell_noobj_loss = self.mse(pred_conf, target_conf)
                        noobj_loss += cell_noobj_loss
                        total_loss += self.lambda_noobj * cell_noobj_loss
                
                # Calculate class loss
                class_mask = target_cell[:, 4] == 1
                if class_mask.any():
                    pred_class = pred_cell[class_mask, self.B*5:]
                    target_class = target_cell[class_mask, self.B*5:]
                    cell_class_loss = self.bce(pred_class, target_class)
                    class_loss += cell_class_loss
                    total_loss += self.lambda_class * cell_class_loss
        
        # Normalize losses by batch size
        total_loss = total_loss / batch_size
        coord_loss = coord_loss / batch_size
        obj_loss = obj_loss / batch_size
        noobj_loss = noobj_loss / batch_size
        class_loss = class_loss / batch_size
        
        # Return total loss and components
        loss_components = {
            'coord_loss': coord_loss,
            'obj_loss': obj_loss,
            'noobj_loss': noobj_loss,
            'class_loss': class_loss
        }
        
        return total_loss, loss_components
    
    def _calculate_iou(self, pred_box, target_box):
        """
        Calculate IoU between predicted and target boxes
        pred_box: [N, 5] (x, y, w, h, conf)
        target_box: [N, 5] (x, y, w, h, conf)
        """
        # Extract coordinates
        pred_x = pred_box[:, 0]
        pred_y = pred_box[:, 1]
        pred_w = pred_box[:, 2]
        pred_h = pred_box[:, 3]
        
        target_x = target_box[:, 0]
        target_y = target_box[:, 1]
        target_w = target_box[:, 2]
        target_h = target_box[:, 3]
        
        # Calculate intersection area
        w_intersection = torch.min(pred_w, target_w)
        h_intersection = torch.min(pred_h, target_h)
        intersection = w_intersection * h_intersection
        
        # Calculate union area
        pred_area = pred_w * pred_h
        target_area = target_w * target_h
        union = pred_area + target_area - intersection
        
        # Calculate IoU
        iou = intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero
        
        return iou 