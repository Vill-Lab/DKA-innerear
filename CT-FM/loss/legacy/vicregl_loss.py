import lightly.loss.vicregl_loss
from torch import Tensor


class VICRegLLoss(lightly.loss.vicregl_loss.VICRegLLoss):

    # IMPORTANT: this is a method that was changed
    def forward(self, out):
        global_views_features, global_grids, local_views_features, local_grids = out
        return super().forward(global_views_features, global_grids, local_views_features, local_grids)

    def _local_l2_loss(
        self,
        z_a: Tensor,
        z_b: Tensor,
    ) -> Tensor:
        """Returns loss for local features matched with neareast neighbors using L2
        distance in the feature space.

        Args:
            z_a:
                Local feature tensor with shape (batch_size, heigh, width, dim).
            z_b:
                Local feature tensor with shape (batch_size, heigh, width, dim).
        """
        # (batch_size, heigh, width, dim) -> (batch_size, heigh * width, dim)
        z_a = z_a.flatten(start_dim=1, end_dim=(z_a.ndim - 2))  # IMPORTANT: this is a line that was changed
        z_b = z_b.flatten(start_dim=1, end_dim=(z_b.ndim - 2))  # IMPORTANT: this is a line that was changed

        z_a_filtered, z_a_nn = self._nearest_neighbors_on_l2(
            input_features=z_a, candidate_features=z_b, num_matches=self.num_matches[0]
        )
        z_b_filtered, z_b_nn = self._nearest_neighbors_on_l2(
            input_features=z_b, candidate_features=z_a, num_matches=self.num_matches[1]
        )
        loss_a = self.vicreg_loss.forward(z_a=z_a_filtered, z_b=z_a_nn)
        loss_b = self.vicreg_loss.forward(z_a=z_b_filtered, z_b=z_b_nn)
        return 0.5 * (loss_a + loss_b)

    def _local_location_loss(
        self,
        z_a: Tensor,
        z_b: Tensor,
        grid_a: Tensor,
        grid_b: Tensor,
    ) -> Tensor:
        """Returns loss for local features matched with nearest neighbors based on
        the feature location.

        Args:
            z_a:
                Local feature tensor with shape (batch_size, heigh, width, dim).
            z_b:
                Local feature tensor with shape (batch_size, heigh, width, dim).
                Note that height and width can be different from z_a.
            grid_a:
                Grid tensor with shape (batch_size, height, width, 2).
            grid_b:
                Grid tensor with shape (batch_size, height, width, 2).
                Note that height and width can be different from grid_a.
        """
        # (batch_size, heigh, width, dim) -> (batch_size, heigh * width, dim)
        z_a = z_a.flatten(start_dim=1, end_dim=(z_a.ndim - 2))  # IMPORTANT: this is a line that was changed
        z_b = z_b.flatten(start_dim=1, end_dim=(z_b.ndim - 2))  # IMPORTANT: this is a line that was changed
        # (batch_size, heigh, width, 2) -> (batch_size, heigh * width, 2)
        grid_a = grid_a.flatten(start_dim=1, end_dim=(grid_a.ndim - 2))  # IMPORTANT: this is a line that was changed
        grid_b = grid_b.flatten(start_dim=1, end_dim=(grid_b.ndim - 2))  # IMPORTANT: this is a line that was changed
        z_a_filtered, z_a_nn = self._nearest_neighbors_on_grid(
            input_features=z_a,
            candidate_features=z_b,
            input_grid=grid_a,
            candidate_grid=grid_b,
            num_matches=self.num_matches[0],
        )
        z_b_filtered, z_b_nn = self._nearest_neighbors_on_grid(
            input_features=z_b,
            candidate_features=z_a,
            input_grid=grid_b,
            candidate_grid=grid_a,
            num_matches=self.num_matches[1],
        )

        loss_a = self.vicreg_loss.forward(z_a=z_a_filtered, z_b=z_a_nn)
        loss_b = self.vicreg_loss.forward(z_a=z_b_filtered, z_b=z_b_nn)
        return 0.5 * (loss_a + loss_b)
