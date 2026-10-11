import numpy as np

class ZeroCopyBatchLoader:
    def __init__(self, data: np.ndarray, batch_size: int):
        """Store data in a flat contiguous buffer simulating shared memory."""
        self.batch_size=batch_size
        self.size=data.shape
        self.data=data.flatten()
        
        


    def num_batches(self) -> int:
        """Return total number of batches."""
        return int((self.size[0]+self.batch_size-1)/self.batch_size)

    def get_batch(self, batch_idx: int) -> np.ndarray:
        """Return batch as a zero-copy view into the buffer."""
        
        elements_per_batch=self.batch_size*self.size[1]
        return self.data[(elements_per_batch*batch_idx):(elements_per_batch*(batch_idx+1))].reshape(-1,self.size[1])

    def is_zero_copy(self, batch_idx: int) -> bool:
        """Check whether the batch shares memory with the buffer."""
        
        return np.shares_memory(self.data,self.get_batch(batch_idx))

    def get_batch_means(self) -> list:
        """Return list of per-batch mean values, each rounded to 4 decimals."""
        mean_list=[]
        
        for i in range(self.num_batches()):

            one_bmean=round(float(self.get_batch(i).mean()),4)
            mean_list.append(one_bmean)
        return mean_list

    def write_to_buffer(self, row: int, col: int, value: float) -> None:
        """Write a value directly into the flat buffer at (row, col)."""
        self.data[row*self.size[1]+col]=value
        return self.data.reshape(self.size)