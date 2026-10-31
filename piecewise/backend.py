# Piecewise torch.compile backend'i: graph'ı attention noktalarından böler, attention içermeyen
# parçaları inductor ile derleyip CUDA graph olarak capture eder, gather + attention gruplarını eager bırakır.
import torch
from torch._inductor.compile_fx import compile_fx

from piecewise.utils import create_partition, fake_inputs, is_graph_piece, split, weak_ref


def capture(forward, pool, *args):
    with torch.inference_mode():
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            forward(*args)

        torch.cuda.current_stream().wait_stream(side_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            output = forward(*args)

        graph.replay()

    return graph, output


class Graph:

    def __init__(self, graph: torch.cuda.CUDAGraph, output) -> None:
        self.graph = graph
        self.output = weak_ref(output)

    def __call__(self, *args):
        self.graph.replay()
        return self.output


def buffered(submod, out: torch.Tensor):
    def run(*args):
        out.copy_(submod(*args))
        return out
    return run


def lazy_graph(split_gm, name: str, submod, pool, graphs, key):
    def create(*args):
        f_inputs = fake_inputs(submod)

        compiled = compile_fx(submod, f_inputs)
        graph, output = capture(compiled, pool, *args)
        obj = Graph(graph, output)

        split_gm.__dict__[name] = obj
        graphs[key] = obj
        return output
    return create


def compile_pieces(split_gm, out: torch.Tensor, pool, graphs) -> None:
    for name, submod in list(split_gm.named_children()):
        if is_graph_piece(submod):

            key = (name, submod.code, tuple(
                (str(v.shape), str(v.stride()), v.dtype, v.device)
                if isinstance(v, torch.Tensor)
                else repr(v)
                for v in fake_inputs(submod)
            ))

            if key not in graphs:
                graph = lazy_graph(
                    split_gm, name,
                    submod, pool,
                    graphs, key
                )
            else:
                graph = graphs[key]

            split_gm.__dict__[name] = graph
        else:
            split_gm.__dict__[name] = buffered(submod, out)


def get_piecewise_backend(out: torch.Tensor, pool):
    graphs = {}
    def piecewise_backend(gm, example_inputs):
        split_gm = split(gm, create_partition(gm))
        compile_pieces(split_gm, out, pool, graphs)
        return split_gm
    return piecewise_backend
