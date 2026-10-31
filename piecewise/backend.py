# Piecewise torch.compile backend'i: graph'ı attention noktalarından böler, attention içermeyen
# parçaları inductor ile derleyip CUDA graph olarak capture eder, gather + attention gruplarını eager bırakır.
import torch
from torch._inductor.compile_fx import compile_fx

from piecewise.utils import create_partition, fake_inputs, is_graph_piece, split


class Graph:

    def __init__(self, forward, *args) -> None:
        with torch.inference_mode():
            side_stream = torch.cuda.Stream()
            side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side_stream):
                forward(*args)

            torch.cuda.current_stream().wait_stream(side_stream)

            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = forward(*args)

    def __call__(self, *args):
        self.graph.replay()
        return self.output


def buffered(submod, out: torch.Tensor):
    def run(*args):
        out.copy_(submod(*args))
        return out
    return run


def lazy_graph(split_gm, name: str, submod):
    def create(*args):
        compiled = compile_fx(submod, fake_inputs(submod))
        graph = Graph(compiled, *args)
        split_gm.__dict__[name] = graph
        return graph(*args)
    return create


def compile_pieces(split_gm, out: torch.Tensor) -> None:
    for name, submod in list(split_gm.named_children()):
        if is_graph_piece(submod):
            split_gm.__dict__[name] = lazy_graph(split_gm, name, submod)
        else:
            split_gm.__dict__[name] = buffered(submod, out)


def get_piecewise_backend(out: torch.Tensor):
    def piecewise_backend(gm, example_inputs):
        split_gm = split(gm, create_partition(gm))
        compile_pieces(split_gm, out)
        return split_gm
    return piecewise_backend
