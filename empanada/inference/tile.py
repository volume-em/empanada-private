import math
import numpy as np
from cztile.fixed_total_area_strategy import AlmostEqualBorderFixedTotalAreaStrategy2D
from cztile.tiling_strategy import Rectangle as czrect
from empanada.array_utils import rle_voting, merge_rles

__all__ = ['Tiler', 'Cuber']

def calculate_overlap_rle(yranges, xranges, image_shape):
    r"""Creates a run length encoding of the overlap between tiles.

    Args:
        yranges (list): List of tuples of the form (ymin, ymax)
        xranges (list): List of tuples of the form (xmin, xmax)
        image_shape (tuple): Shape of the image (h, w)

    Returns:
        starts: Array of (n,), the run length encoding starts.
        runs: Array of (n,), the run length encoding runs.

    """
    y = np.array(
        rle_voting(np.unique(np.stack(yranges, axis=0), axis=0), vote_thr=2)
    )
    x = np.array(
        rle_voting(np.unique(np.stack(xranges, axis=0), axis=0), vote_thr=2)
    )
    
    if len(y) > 0:
        row_s = y[:, 0]
        row_e = y[:, 1]
        row_starts = row_s * image_shape[1]
        row_runs = row_e * image_shape[1] - row_starts
    else:
        row_starts = []
        row_runs = []
    
    if len(x) > 0:
        col_ranges = []
        for r in range(image_shape[0]):
            col_ranges.append(x + r * image_shape[1])

        col_ranges = np.concatenate(col_ranges, axis=0)
        col_starts = col_ranges[:, 0]
        col_runs = col_ranges[:, 1] - col_starts
    else:
        col_starts = []
        col_runs = []

    if len(row_starts) > 0 or len(col_starts) > 0:
        return merge_rles(row_starts, row_runs, col_starts, col_runs)
    else:
        return [], []

class Tiler:
    def __init__(
        self, 
        image_shape, 
        tile_size=2048, 
        overlap_width=128
    ):
        r"""Creates a tiling strategy for a given image shape.

        Args:
            image_shape: Tuple of image (height, width). Must be 2D.

            tile_size: Integer or Tuple. Size of the tile in pixels.
                If an integer, the tiles will be square.

            overlap_width: Integer. Minimum number of pixels to overlap between tiles.

        """
        if isinstance(tile_size, int):
            tile_size = (tile_size, tile_size)
            
        assert isinstance(overlap_width, int)
        assert len(image_shape) == 2, "Tiler only works with 2D images"

        self.image_shape = image_shape
        self.tile_size = tile_size
        self.overlap_width = overlap_width

        th, tw = tile_size
        
        # adjust the tile size to fit into the image shape
        th = min(th, image_shape[0])
        tw = min(tw, image_shape[1])

        tiler = AlmostEqualBorderFixedTotalAreaStrategy2D(
            total_tile_width=tw, total_tile_height=th, 
            min_border_width=overlap_width
        )

        h, w = image_shape
        rectangle = czrect(x=0, y=0, w=w, h=h)

        # define the y and x ranges of each tile
        yranges = []
        xranges = []
        for tile in tiler.tile_rectangle(rectangle):
            y, x = tile.roi.y, tile.roi.x
            h, w = tile.roi.h, tile.roi.w

            yranges.append((y, y + h))
            xranges.append((x, x + w))

        self.overlap_rle = calculate_overlap_rle(yranges, xranges, image_shape)
        self.yranges = yranges
        self.xranges = xranges

    def __len__(self):
        return len(self.yranges)

    def overlap_mask(self):
        r"""Returns an array showing the overlapping areas between tiles.
        """
        overlap = np.zeros(np.prod(self.image_shape))
        for s,r in zip(self.overlap_rle[0], self.overlap_rle[1]):
            overlap[s:s + r] = 1

        return overlap.reshape(self.image_shape)

    def translate_rle_seg(self, rle_seg, tile_index):
        r"""Translates the bounding box and run length encoding
        start indices from the tile coordinate frame to the
        global coordinate frame.

        Args:
            rle_seg: A run length encoded segmentation mask like the
                output of pan_seg_to_rle_seg.

            tile_index: Integer. The index of the tile represented by
                the rle_seg. Must be within length of this tiler.

        Returns:
            translated_rle_seg: The run length encoded segmentation mask
                with translated instance boxes and rles. Note, that the
                translation happens inplace.

        """
        # get position of the tile from index
        ys, ye = self.yranges[tile_index]
        xs, xe = self.xranges[tile_index]
        h, w = ye - ys, xe - xs

        # loop over each class and all labels in the rle_seg
        for class_id, labels in rle_seg.items():
            for label, label_attrs in labels.items():
                # shift bounding box into the tile frame
                shifted_box = list(label_attrs['box'])
                shifted_box[0] += ys
                shifted_box[1] += xs
                shifted_box[2] += ys
                shifted_box[3] += xs
                label_attrs['box'] = tuple(shifted_box)
                
                # shift the rle starting indices into the tile frame
                starts = label_attrs['starts']
                starts_y = starts // w
                starts_x = starts % w
                starts_y += ys
                starts_x += xs

                # get starts are translated raveled indices
                label_attrs['starts'] = np.ravel_multi_index(
                    (starts_y, starts_x), dims=self.image_shape
                ) 

        return rle_seg
    
    def get_tile_box(self, tile_index):
        ys, ye = self.yranges[tile_index]
        xs, xe = self.xranges[tile_index]
        
        return [ys, xs, ye, xe]

    def __call__(self, image, tile_index):
        r"""Crops the given image into a particular tile.

        Args:
            image: Array of (h, w).

            tile_index: Integer. Index of the tile to crop. 
                Must be within length of this tiler.

        Returns:
            tile_image: Array of tile_size.

        """
        if tile_index >= len(self):
            raise IndexError("Tile index out of range")
        else:
            tile_index = tile_index % len(self)

        assert image.shape == self.image_shape, \
        "Image shape of {image.shape} does not match tiler expected shape {self.image_shape}"
        
        # get the slices into the image
        yslice = slice(*self.yranges[tile_index])
        xslice = slice(*self.xranges[tile_index])

        return image[yslice, xslice]
    
class Cuber:
    def __init__(self, array_shape, cube_shape, halo=0.1):
        assert len(array_shape) == len(cube_shape) == 3
        self.array_shape = array_shape
        self.cube_shape = cube_shape
        self.halo = tuple([int(s * halo) for s in cube_shape])
        
        self.chunk_dims = tuple(
            [math.ceil(s / cs) for s,cs in zip(array_shape, cube_shape)]
        )
        
        self.cubes = self._get_cubes()
        
    def _get_cubes(self):
        r"""Create slicable data cubes for the given array.
        
        Returns:
            cubes (Dict[int, Tuple(slice)]): cube ROIs indexed
            by the raveled cube index
        
        """
        cubes = {}

        d, h, w = self.array_shape
        zs, ys, xs = self.cube_shape
        cd, ch, cw = self.chunk_dims
        
        for zc, z in enumerate(range(0, d, zs)):
            for yc, y in enumerate(range(0, h, ys)):
                for xc, x in enumerate(range(0, w, xs)):
                    global_slices = (
                        slice(z, min(z + zs, d)),
                        slice(y, min(y + ys, h)),
                        slice(x, min(x + xs, w))
                    )
                    
                    global_halo = tuple([
                        slice(max(0, sl.start - h), min(s, sl.stop + h))
                        for h,s,sl in zip(self.halo, self.array_shape, global_slices)
                    ])
                    
                    local_slices = tuple([
                        slice(gs.start - hs.start, (gs.start - hs.start) + (gs.stop - gs.start))
                        for gs,hs in zip(global_slices, global_halo)
                    ])
                    
                    # compute the raveled cube index
                    cube_index = (zc * ch * cw) + (yc * cw) + xc
                    cubes[cube_index] = {
                        'fill': global_slices,
                        'cut': local_slices,
                        'infer': global_halo
                    }
                    
        return cubes
    
    def find_neighbors(self, cube_index):
        r"""Finds the indices of cubes to the right, bottom and back
        of the given cube.
        
        Args:
            cube_index (int): Index of a cube in the Cuber.
            
        Returns:
            neighbors (Tuple[int]): Index of cube neighbors
            to the right, bottom and back respectively. Any or
            all of the neighbors may be None.
        
        """
        cd, ch, cw = self.chunk_dims
        
        # get the raveled cube indices
        if (cube_index + 1) % cw != 0:
            right = cube_index + 1
        else:
            right = None
            
        if (cube_index // cw + 1) % ch != 0:
            bottom = cube_index + cw
        else:
            bottom = None
            
        if (cube_index // (ch * cw) + 1) % cd != 0:
            back = cube_index + (ch * cw)
        else:
            back = None
        
        return (right, bottom, back)
        
        