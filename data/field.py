# coding: utf8
from collections import Counter, OrderedDict
from torch.utils.data.dataloader import default_collate
from itertools import chain
import six
import torch
import numpy as np
import h5py
import os
import warnings
import shutil
from pathlib import Path
from PIL import Image

from .dataset import Dataset
from .vocab import Vocab
from .utils import get_tokenizer
import clip


class RawField(object):
    """ Defines a general datatype.

    Every dataset consists of one or more types of data. For instance,
    a machine translation dataset contains paired examples of text, while
    an image captioning dataset contains images and texts.
    Each of these types of data is represented by a RawField object.
    An RawField object does not assume any property of the data type and
    it holds parameters relating to how a datatype should be processed.

    Attributes:
        preprocessing: The Pipeline that will be applied to examples
            using this field before creating an example.
            Default: None.
        postprocessing: A Pipeline that will be applied to a list of examples
            using this field before assigning to a batch.
            Function signature: (batch(list)) -> object
            Default: None.
    """

    def __init__(self, preprocessing=None, postprocessing=None):
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

    def preprocess(self, x):
        """ Preprocess an example if the `preprocessing` Pipeline is provided. """
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, *args, **kwargs):
        """ Process a list of examples to create a batch.

        Postprocess the batch with user-provided Pipeline.

        Args:
            batch (list(object)): A list of object from a batch of examples.
        Returns:
            object: Processed object given the input and custom
                postprocessing Pipeline.
        """
        if self.postprocessing is not None:
            batch = self.postprocessing(batch)
        return default_collate(batch)


class Merge(RawField):
    def __init__(self, *fields):
        super(Merge, self).__init__()
        self.fields = fields

    def preprocess(self, x):
        return tuple(f.preprocess(x) for f in self.fields)

    def process(self, batch, *args, **kwargs):
        if len(self.fields) == 1:
            batch = [batch, ]
        else:
            batch = list(zip(*batch))

        out = list(f.process(b, *args, **kwargs) for f, b in zip(self.fields, batch))
        return out

class ImageDetectionsField(RawField):
    def __init__(self, preprocessing=None, postprocessing=None, detections_path=None, max_detections=1000,
                 sort_by_prob=False, load_in_tmp=True):
        self.detections_path = detections_path
        self.sort_by_prob = sort_by_prob
        self._hdf5_file = None
        self._hdf5_index = None
        self._hdf5_mode = None
        self._is_hdf5 = self._detect_hdf5_path(detections_path)

        super(ImageDetectionsField, self).__init__(preprocessing, postprocessing)

    def preprocess(self, x, avoid_precomp=False):
        image_name = os.path.splitext(os.path.basename(x))[0]
        if self._is_hdf5:
            features = self._load_from_hdf5(image_name)
        else:
            features = np.load(os.path.join(self.detections_path, f"{image_name}.npy"), allow_pickle=False)
        return features.astype(np.float32)

    def _detect_hdf5_path(self, path):
        if path is None:
            return False
        if isinstance(path, os.PathLike):
            path = os.fspath(path)
        return os.path.isfile(path) and path.lower().endswith((".h5", ".hdf5"))

    def _ensure_hdf5(self):
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.detections_path, 'r')
            
            # Check if it is monolithic (Swin) or key-value (X152)
            if 'filenames' in self._hdf5_file and 'features' in self._hdf5_file:
                self._hdf5_mode = 'monolithic'
                
                cache_path = self.detections_path + ".index.pkl"
                if os.path.exists(cache_path):
                    import pickle
                    with open(cache_path, 'rb') as f:
                        self._hdf5_index = pickle.load(f)
                else:
                    filenames = self._hdf5_file['filenames'][...]
                    self._hdf5_index = {
                        (name.decode('utf-8') if isinstance(name, bytes) else str(name)): idx
                        for idx, name in enumerate(filenames)
                    }
                    try:
                        import pickle
                        with open(cache_path, 'wb') as f:
                            pickle.dump(self._hdf5_index, f)
                    except Exception:
                        pass
            else:
                self._hdf5_mode = 'key-value'

    def _load_from_hdf5(self, image_name: str) -> np.ndarray:
        self._ensure_hdf5()
        
        if self._hdf5_mode == 'monolithic':
            if image_name not in self._hdf5_index:
                raise KeyError(f"Image id '{image_name}' not found in HDF5 features at {self.detections_path}")
            idx = self._hdf5_index[image_name]
            return self._hdf5_file['features'][idx]
        
        else: # key-value mode
            # image_name format: COCO_train2014_000000000009
            try:
                image_id = int(image_name.split('_')[-1])
                key = f"{image_id}_grids"
            except Exception:
                key = f"{image_name}_grids"

            if key in self._hdf5_file:
                return self._hdf5_file[key][()]
            else:
                # Fallback or error
                raise KeyError(f"Key '{key}' (derived from '{image_name}') not found in HDF5 file.")

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_hdf5_file'] = None
        return state

    def close(self):
        if self._hdf5_file is not None:
            try:
                self._hdf5_file.close()
            finally:
                self._hdf5_file = None

    def __del__(self):
        self.close()

class ImageDetectionsField_bak(RawField):
    def __init__(self, preprocessing=None, postprocessing=None, detections_path=None, max_detections=100,
                 sort_by_prob=False, load_in_tmp=True):
        self.max_detections = max_detections
        self.detections_path = detections_path
        self.sort_by_prob = sort_by_prob

        tmp_detections_path = os.path.join('/tmp', os.path.basename(detections_path))

        if load_in_tmp:
            if not os.path.isfile(tmp_detections_path):
                if shutil.disk_usage("/tmp")[-1] < os.path.getsize(detections_path):
                    warnings.warn('Loading from %s, because /tmp has no enough space.' % detections_path)
                else:
                    warnings.warn("Copying detection file to /tmp")
                    shutil.copyfile(detections_path, tmp_detections_path)
                    warnings.warn("Done.")
                    self.detections_path = tmp_detections_path
            else:
                self.detections_path = tmp_detections_path

        super(ImageDetectionsField, self).__init__(preprocessing, postprocessing)

    def preprocess(self, x, avoid_precomp=False):
        image_id = int(x.split('_')[-1].split('.')[0])
        try:
            f = h5py.File(self.detections_path, 'r')
#            precomp_data = f['%d_features' % image_id][()]
            precomp_data = f['%d_grids' % image_id][()]
            if self.sort_by_prob:
                precomp_data = precomp_data[np.argsort(np.max(f['%d_cls_prob' % image_id][()], -1))[::-1]]
        except KeyError:
            warnings.warn('Could not find detections for %d' % image_id)
            precomp_data = np.random.rand(10,2048)

        delta = self.max_detections - precomp_data.shape[0]
        if delta > 0:
            precomp_data = np.concatenate([precomp_data, np.zeros((delta, precomp_data.shape[1]))], axis=0)
        elif delta < 0:
            precomp_data = precomp_data[:self.max_detections]

        return precomp_data.astype(np.float32)

def build_transform(is_train, config):
    model_name = getattr(config, 'CLIP_MODEL_NAME', "ViT-L/14")
    _, preprocess = clip.load(model_name, device='cpu')
    return preprocess


class ImageField(RawField):
    def __init__(self, preprocessing=None, postprocessing=None, config=None):
        self.config = config
        self.train_transform = build_transform(True, config)
        self.test_transform = build_transform(False, config)
        super(ImageField, self).__init__(preprocessing, postprocessing)

    def preprocess(self, x, is_train=False):
        image_path = x
        if not os.path.exists(image_path):
            if 'images' in x:
                relative_part = x.split('images')[-1].lstrip('/')
                image_path = os.path.join(self.config.img_root_path, relative_part)
            else:
                image_path = os.path.join(self.config.img_root_path, x)
        img = Image.open(image_path).convert('RGB')
        transform = self.train_transform if is_train else self.test_transform
        return transform(img)

class TextField(RawField):
    vocab_cls = Vocab
    # Dictionary mapping PyTorch tensor dtypes to the appropriate Python
    # numeric type.
    dtypes = {
        torch.float32: float,
        torch.float: float,
        torch.float64: float,
        torch.double: float,
        torch.float16: float,
        torch.half: float,

        torch.uint8: int,
        torch.int8: int,
        torch.int16: int,
        torch.short: int,
        torch.int32: int,
        torch.int: int,
        torch.int64: int,
        torch.long: int,
    }
    punctuations = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
                    ".", "?", "!", ",", ":", "-", "--", "...", ";"]

    def __init__(self, use_vocab=True, init_token=None, eos_token=None, fix_length=None, dtype=torch.long,
                 preprocessing=None, postprocessing=None, lower=False, tokenize=(lambda s: s.split()),
                 remove_punctuation=False, include_lengths=False, batch_first=True, pad_token="<pad>",
                 unk_token="<unk>", pad_first=False, truncate_first=False, vectors=None, nopoints=True):
        self.use_vocab = use_vocab
        self.init_token = init_token
        self.eos_token = eos_token
        self.fix_length = fix_length
        self.dtype = dtype
        self.lower = lower
        self.tokenize = get_tokenizer(tokenize)
        self.remove_punctuation = remove_punctuation
        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_first = pad_first
        self.truncate_first = truncate_first
        self.vocab = None
        self.vectors = vectors
        if nopoints:
            self.punctuations.append("..")

        super(TextField, self).__init__(preprocessing, postprocessing)

    def preprocess(self, x):
        if six.PY2 and isinstance(x, six.string_types) and not isinstance(x, six.text_type):
            x = six.text_type(x, encoding='utf-8')
        if self.lower:
            x = six.text_type.lower(x)
        x = self.tokenize(x.rstrip('\n'))
        if self.remove_punctuation:
            x = [w for w in x if w not in self.punctuations]
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, device=None):
        padded = self.pad(batch)
        tensor = self.numericalize(padded, device=device)
        return tensor

    def build_vocab(self, *args, **kwargs):
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in arg.fields.items() if field is self]
            else:
                sources.append(arg)

        for data in sources:
            for x in data:
                x = self.preprocess(x)
                try:
                    counter.update(x)
                except TypeError:
                    counter.update(chain.from_iterable(x))

        specials = list(OrderedDict.fromkeys([
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token]
            if tok is not None]))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

    def pad(self, minibatch):
        """Pad a batch of examples using this field.
        Pads to self.fix_length if provided, otherwise pads to the length of
        the longest example in the batch. Prepends self.init_token and appends
        self.eos_token if those attributes are not None. Returns a tuple of the
        padded list and a list containing lengths of each example if
        `self.include_lengths` is `True`, else just
        returns the padded list.
        """
        minibatch = list(minibatch)
        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = self.fix_length + (
                self.init_token, self.eos_token).count(None) - 2
        padded, lengths = [], []
        for x in minibatch:
            if self.pad_first:
                padded.append(
                    [self.pad_token] * max(0, max_len - len(x)) +
                    ([] if self.init_token is None else [self.init_token]) +
                    list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                    ([] if self.eos_token is None else [self.eos_token]))
            else:
                padded.append(
                    ([] if self.init_token is None else [self.init_token]) +
                    list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                    ([] if self.eos_token is None else [self.eos_token]) +
                    [self.pad_token] * max(0, max_len - len(x)))
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        if self.include_lengths:
            return padded, lengths
        return padded

    def numericalize(self, arr, device=None):
        """Turn a batch of examples that use this field into a list of Variables.
        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.
        Arguments:
            arr (List[List[str]], or tuple of (List[List[str]], List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (str or torch.device): A string or instance of `torch.device`
                specifying which device the Variables are going to be created on.
                If left as default, the tensors will be created on cpu. Default: None.
        """
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.tensor(lengths, dtype=self.dtype, device=device)

        if self.use_vocab:
            arr = [[self.vocab.stoi[x] for x in ex] for ex in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, self.vocab)

            var = torch.tensor(arr, dtype=self.dtype, device=device)
        else:
            if self.vectors:
                arr = [[self.vectors[x] for x in ex] for ex in arr]
            if self.dtype not in self.dtypes:
                raise ValueError(
                    "Specified Field dtype {} can not be used with "
                    "use_vocab=False because we do not know how to numericalize it. "
                    "Please raise an issue at "
                    "https://github.com/pytorch/text/issues".format(self.dtype))
            numericalization_func = self.dtypes[self.dtype]
            # It doesn't make sense to explictly coerce to a numeric type if
            # the data is sequential, since it's unclear how to coerce padding tokens
            # to a numeric type.
            arr = [numericalization_func(x) if isinstance(x, six.string_types)
                   else x for x in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, None)

            var = torch.cat([torch.cat([a.unsqueeze(0) for a in ar]).unsqueeze(0) for ar in arr])

        # var = torch.tensor(arr, dtype=self.dtype, device=device)
        if not self.batch_first:
            var.t_()
        var = var.contiguous()

        if self.include_lengths:
            return var, lengths
        return var

    def decode(self, word_idxs, join_words=True):
        if isinstance(word_idxs, list) and len(word_idxs) == 0:
            return self.decode([word_idxs, ], join_words)[0]
        if isinstance(word_idxs, list) and isinstance(word_idxs[0], int):
            return self.decode([word_idxs, ], join_words)[0]
        elif isinstance(word_idxs, np.ndarray) and word_idxs.ndim == 1:
            return self.decode(word_idxs.reshape((1, -1)), join_words)[0]
        elif isinstance(word_idxs, torch.Tensor) and word_idxs.ndimension() == 1:
            return self.decode(word_idxs.unsqueeze(0), join_words)[0]

        captions = []
        for wis in word_idxs:
            caption = []
            for wi in wis:
                word = self.vocab.itos[int(wi)]
                if word == self.eos_token:
                    break
                caption.append(word)
            if join_words:
                caption = ' '.join(caption)
            captions.append(caption)
        return captions


class PreExtractedCLIPField(RawField):
    def __init__(self, features_path=None, h5_train_path=None, h5_val_path=None, h5_default_path=None):
        super().__init__()
        self.features_path = features_path
        base = Path(features_path) if features_path else None
        self._h5_index_cache = {}

        # Auto-detect common merged filenames if explicit paths not provided
        if base and h5_train_path is None:
            candidate = base / "merged_train2014.h5"
            if candidate.exists():
                h5_train_path = str(candidate)
        if base and h5_val_path is None:
            candidate = base / "merged_val2014.h5"
            if candidate.exists():
                h5_val_path = str(candidate)
        if base and h5_default_path is None:
            candidate = base / "merged_features.h5"
            if candidate.exists():
                h5_default_path = str(candidate)

        self.h5_train_path = h5_train_path
        self.h5_val_path = h5_val_path
        self.h5_default_path = h5_default_path
        self._h5_handles = {}

    def _load_from_h5(self, h5_path: str, key: str) -> np.ndarray:
        if h5_path not in self._h5_handles:
            h5_file = h5py.File(h5_path, "r")

            if h5_path in self._h5_index_cache:
                index = self._h5_index_cache[h5_path]
            else:
                names = h5_file["filenames"][...]
                names = [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in names]
                index = {n: i for i, n in enumerate(names)}
                self._h5_index_cache[h5_path] = index

            self._h5_handles[h5_path] = (h5_file, index)

        h5_file, index = self._h5_handles[h5_path]
        if key not in index:
            raise KeyError(f"{key} not found in {h5_path}")
        return h5_file["features"][index[key]][()]

    def _select_h5(self, image_path: str) -> str | None:
        if self.h5_train_path and "train2014" in image_path:
            return self.h5_train_path
        if self.h5_val_path and "val2014" in image_path:
            return self.h5_val_path
        return self.h5_default_path

    def preprocess(self, x):
        stem = os.path.splitext(os.path.basename(x))[0]
        h5_path = self._select_h5(x)

        if h5_path:
            features = self._load_from_h5(h5_path, stem)
            return np.asarray(features, dtype=np.float32)

        npy_path = os.path.splitext(x)[0] + '.npy'
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"Pre-extracted features not found: {npy_path}")
        features = np.load(npy_path)
        return features.astype(np.float32)

    def process(self, batch):
        tensor_batch = [torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr for arr in batch]
        return torch.stack(tensor_batch, dim=0)

    def close(self):
        for h5_file, _ in self._h5_handles.values():
            try:
                h5_file.close()
            except Exception:
                pass
        self._h5_handles = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_h5_handles'] = {}
        return state

    def __del__(self):
        self.close()