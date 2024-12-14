
import numpy as np
import skimage.transform
import sklearn.metrics

class SegEval:
    """
    This class evaluates a segmented mask agains the ground truth mask and computes all segmentation metrics.
    The metrics are computed in the __init__ function and there are functions describing the aliases of the various
    performance values.
    """

    def __init__(self, mask_pred_bin : np.ndarray, mask_true : np.ndarray, mask_pred_prob : np.ndarray = None, threshold : float = 0.5) -> None:
        """
        Initialisation computes all performance metrics between the true mask and the ground truth mask.
        The function resizes the prediction mask to have the same shape as the true mask - it is possible
        that there is a difference of a couple pixels but there shouldn't be a large difference.
        The binary masks (form which metrics are computed) are acquired by threhsolding with 'threshold'.

        Arguments:
        - 'mask_pred_bin' - The predicted binary mask.
        - 'mask_true' - The true mask.
        - 'mask_pred_prob' - The predicted probability mask if available.
        - 'threhsold' - Threshold used to compute the binary masks from the input.
        """
        self.threshold = threshold
        self.mask_true = np.squeeze(mask_true)
        self.mask_true_bin = self.mask_true >= self.threshold
        self.mask_pred_bin = np.squeeze(mask_pred_bin)
        self.mask_pred_resized = skimage.transform.resize(self.mask_pred_bin, self.mask_true.shape)
        self.mask_pred_bin = self.mask_pred_resized >= self.threshold
        if mask_pred_prob is not None:
            self.mask_pred_prob = np.squeeze(mask_pred_prob)
            self.mask_pred_prob = skimage.transform.resize(self.mask_pred_prob, self.mask_true.shape)
        else:
            self.mask_pred_prob = self.mask_pred_bin

        self.intersection = np.logical_and(self.mask_pred_bin, self.mask_true_bin)
        self.union = np.logical_or(self.mask_pred_bin, self.mask_true_bin)
        self.tp = np.logical_and(self.mask_pred_bin, self.mask_true_bin)
        self.tn = np.logical_and(np.logical_not(self.mask_pred_bin), np.logical_not(self.mask_true_bin))
        self.fp = np.logical_and(self.mask_pred_bin, np.logical_not(self.mask_true_bin))
        self.fn = np.logical_and(np.logical_not(self.mask_pred_bin), self.mask_true_bin)

        self.area_pred = np.sum(self.mask_pred_bin)
        self.area_true = np.sum(self.mask_true_bin)
        self.area_intersection = np.sum(self.intersection)
        self.area_union = np.sum(self.union)
        self.area_tp = np.sum(self.tp)
        self.area_tn = np.sum(self.tn)
        self.area_fp = np.sum(self.fp)
        self.area_fn = np.sum(self.fn)

        self.iou_value = self.area_intersection / self.area_union
        self.dice_value = (2 * self.area_intersection) / (self.area_pred + self.area_true)
        self.precision_value = self.area_tp / (self.area_tp + self.area_fp)
        self.sensitivity_value = self.area_tp / (self.area_tp + self.area_fn)
        self.specificity_value = self.area_tn / (self.area_tn + self.area_fp)
        self.accuracy_value = (self.area_tp + self.area_tn) / (self.area_tp + self.area_fn + self.area_tn + self.area_fp)
        self.roc_auc_value = sklearn.metrics.roc_auc_score(self.mask_true_bin.flatten(), self.mask_pred_prob.flatten())

    def iou(self) -> float:
        """Returns the intersection over union value."""
        return self.iou_value

    def jaccard(self) -> float:
        """Returns the Jaccard coefficient (AKA intersection over union)."""
        return self.iou()
    
    def overlapAreaRatio(self) -> float:
        """Returns the overlap area ratio (AKA intersection voer union)."""
        return self.iou()
    
    def nonOverlapAreaRatio(self) -> float:
        """Returns the non-overalp area ratio, which is simply overlap area ratio subtracted from 1."""
        return 1.0 - self.iou()
    
    def diceCoefficient(self) -> float:
        """Returns the dice coefficient (AKA F-score/F1-score)."""
        return self.dice_value

    def f1(self) -> float:
        """Returns the F1 score (AKA dice coefficient)."""
        return self.diceCoefficient()
    
    def fScore(self) -> float:
        """Returns the F-score value (AKA dice coefficient and F1 score)."""
        return self.diceCoefficient()
    
    def precision(self) -> float:
        """Returns the precision."""
        return self.precision_value
    
    def sensitivity(self) -> float:
        """Returns the sensitivity."""
        return self.sensitivity_value

    def recall(self) -> float:
        """Returns the recall (AKA sensitivity)."""
        return self.sensitivity()
    
    def specificity(self) -> float:
        """Returns the specificity."""
        return self.specificity_value

    def accuracy(self) -> float:
        """Returns the accurcay."""
        return self.accuracy_value
    
    def rocAuc(self) -> float:
        """Returns the area under the ROC curve. The value is accurate only if the prediction mask in constructor contained probabilities."""
        return self.roc_auc_value

class SetSegEval:
    """This class makes evaluation of segmentation on a set of data easier by aggregating the result of 'SegEval' evaluator."""

    def __init__(self, threshold = 0.5) -> None:
        """
        Initialises an empty set evaluator. The data are expanded by calling 'addSample' method and statistics can be retreived
        from the class at any point during the evaluation.

        Arguments:
        - 'threshold' - The threshold with which the evaluator converts the bianry mask into boolean masks if they are not in that form already.
        """
        self.threshold = threshold
        self.seg_evals = []

    def addSample(self, mask_pred_bin : np.ndarray, mask_true : np.ndarray, mask_pred_prob : np.ndarray = None) -> None:
        """
        Adds a single sample result of segmentation to the set evaluator. It computes the evaluation
        between the predicted and true masks and stores them in a counter.

        Arguments:
        - 'mask_pred_bin' - The predicted binary mask.
        - 'mask_true' - The true (ground truth) mask.
        - 'mask_pred_prob' - The predicted probability mask - needed for the AUC metric.
        """
        seg_eval = SegEval(mask_pred_bin, mask_true, mask_pred_prob)
        self.seg_evals.append(seg_eval)

    def iouList(self):
        """Returns the list of stored sample intersection over union values."""
        return [sample.iou() for sample in self.seg_evals]
    
    def f1List(self):
        """Returns the list of stored sample F1 values."""
        return [sample.f1() for sample in self.seg_evals]
    
    def recallList(self):
        """Returns the list of stored sample recall values."""
        return [sample.recall() for sample in self.seg_evals]
    
    def precisionList(self):
        """Returns the list of stored sample precision values."""
        return [sample.precision() for sample in self.seg_evals]
    
    def specificityList(self):
        """Returns the list of stored sample specificity values."""
        return [sample.specificity() for sample in self.seg_evals]
    
    def accuracyList(self):
        """Returns the list of stored sample accuracy values."""
        return [sample.accuracy() for sample in self.seg_evals]
    
    def rocAucList(self):
        """Returns the list of stored sample area under curve values for receiver operating characteristics. These values are accurate only if probabilities were supplied during sample adding."""
        return [sample.rocAuc() for sample in self.seg_evals]
    
    def iouMean(self):
        """Returns the mean intersection over union score."""
        return np.mean(self.iouList())
    
    def f1Mean(self):
        """Returns the mean F1 score."""
        return np.mean(self.f1List())
    
    def recallMean(self):
        """Returns the mean recall score."""
        return np.mean(self.recallList())
    
    def precisionMean(self):
        """Returns the mean precision score."""
        return np.mean(self.precisionList())
    
    def specificityMean(self):
        """Returns the mean specificity score."""
        return np.mean(self.specificityList())
    
    def accuracyMean(self):
        """Returns the mean accuracy score."""
        return np.mean(self.accuracyList())
    
    def rocAucMean(self):
        """Returns the mean area under curve score for receiver operating characteristics. The value is accurate only if probabilities were supplied during sample adding."""
        return np.mean(self.rocAucList())
