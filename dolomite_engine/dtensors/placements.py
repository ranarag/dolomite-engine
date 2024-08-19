import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed._tensor._collective_utils import mesh_broadcast
from torch.distributed._tensor.placement_types import Partial, Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh


class DolomitePartialPlacement(Partial):
    def _reduce_value(self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int) -> torch.Tensor:
        # Partial placement contract #1:
        # _reduce_value: reduce the value of the tensor on the mesh dimension
        return funcol.all_reduce(tensor, reduceOp=self.reduce_op, group=(mesh, mesh_dim))

    def _reduce_shard_value(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        shard_spec: Shard,
    ) -> torch.Tensor:
        # Partial placement contract #2:
        # _reduce_shard_value: reduce_scatter the value of the tensor over the mesh dimension
        return shard_spec._reduce_shard_tensor(tensor, mesh, self.reduce_op, mesh_dim)


class DolomiteReplicatePlacement(Replicate):
    def _replicate_tensor(self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int) -> torch.Tensor:
        """
        Replicate (broadcast) a torch.Tensor on a mesh dimension (use
        the first coordinate on the mesh dimension as source of truth)
        """
        my_coordinate = mesh.get_coordinate()
        if my_coordinate is None:
            # if rank is not part of mesh, we simply return an empty tensor
            return tensor.new_empty(0, requires_grad=tensor.requires_grad)

        tensor = tensor.contiguous()
        mesh_broadcast(tensor, mesh, mesh_dim=mesh_dim)
        return tensor
