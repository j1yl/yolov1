import torch
import torch.nn as nn

class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, predictions, targets):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # Calculate IoU for the two predicted boxes with target
        iou_b1 = self.calculate_iou(predictions[..., self.C+1:self.C+5], targets[..., self.C+1:self.C+5])
        iou_b2 = self.calculate_iou(predictions[..., self.C+6:self.C+10], targets[..., self.C+1:self.C+5])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # Get the box with highest IoU
        # bestbox will be indices of 0, 1 for which bbox has highest IoU
        iou_maxes, bestbox = torch.max(ious, dim=0)

        # Create masks
        exists_box = targets[..., self.C:self.C+1]  # Identity of object (1 if exists)

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        # Set boxes with no object in them to 0
        box_predictions = exists_box * (
            bestbox * predictions[..., self.C+6:self.C+10]
            + (1 - bestbox) * predictions[..., self.C+1:self.C+5]
        )

        box_targets = exists_box * targets[..., self.C+1:self.C+5]

        # Take sqrt of width, height
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # (N, S, S, 4) -> (N*S*S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        pred_box = (
            bestbox * predictions[..., self.C+5:self.C+6] +
            (1 - bestbox) * predictions[..., self.C:self.C+1]
        )

        # (N, S, S, 1) -> (N*S*S)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * targets[..., self.C:self.C+1])
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        # (N, S, S, 1) -> (N*S*S)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.C:self.C+1], start_dim=1),
            torch.flatten((1 - exists_box) * targets[..., self.C:self.C+1], start_dim=1)
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.C+5:self.C+6], start_dim=1),
            torch.flatten((1 - exists_box) * targets[..., self.C:self.C+1], start_dim=1)
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        # (N, S, S, 20) -> (N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :self.C], end_dim=-2),
            torch.flatten(exists_box * targets[..., :self.C], end_dim=-2)
        )

        # Sum all losses
        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )
        
        # Return both the total loss and individual components
        loss_components = {
            'coord_loss': self.lambda_coord * box_loss,
            'obj_loss': object_loss,
            'noobj_loss': self.lambda_noobj * no_object_loss,
            'class_loss': class_loss
        }
        
        return loss, loss_components

    def calculate_iou(self, box1, box2):
        """
        Calculate IoU (Intersection over Union) of two bounding boxes.

        Args:
            box1 (tensor): Predicted box of shape (N, S, S, 4)
            box2 (tensor): Target box of shape (N, S, S, 4)

        Returns:
            tensor: IoU value of shape (N, S, S)
        """
        # Convert to (x1, y1, x2, y2) format
        box1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
        box1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
        box1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
        box1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2

        box2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
        box2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
        box2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
        box2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2

        # Calculate intersection area
        x1 = torch.max(box1_x1, box2_x1)
        y1 = torch.max(box1_y1, box2_y1)
        x2 = torch.min(box1_x2, box2_x2)
        y2 = torch.min(box1_y2, box2_y2)

        # Intersection area
        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

        # Union area
        box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
        box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
        union = box1_area + box2_area - intersection + 1e-6

        # IoU
        iou = intersection / union

        return iou 