import torch
import torchvision

def postprocessing(out,nms_thresh=0.5,iou_aware=False) :
	boxes=out[:,:,:4]
	conf=out[:,:,5]

	if not iou_aware :
		cls_sc=out[:,:,5:]
	else :
		cls_sc=out[:,:,5:-1]
		sigma=out[:,:,-1]

	ids=torch.argmax(cls_sc,dim=2)
	scores=conf * torch.max(cls_sc,dim=2)
	if iou_aware :
		scores *= (1-sigma)

	ret_ind=torchvision.ops.batched_nms(boxes,scores,ids,nms_thresh)

	return [boxes[ret_ind],scores[ret_ind],ids[ret_ind]]
