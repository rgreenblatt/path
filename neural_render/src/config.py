import argparse


class Config(argparse.Namespace):
    def ordered_params(self):
        return sorted(vars(self).items())

    def ordered_non_default(self):
        all_defaults = {
            # rebuild parser to avoiding having parser attribute
            key: self.build_parser().get_default(key)
            for key in vars(self)
        }

        def arg_is_default(attr, value):
            return all_defaults[attr] == value

        return map(
            lambda x: x + (all_defaults[x[0]], ),
            filter(lambda x: not arg_is_default(*x), self.ordered_params()))

    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in self.ordered_params():
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def print_non_default(self, prtf=print):
        prtf("")
        prtf("Non default parameters:")
        for attr, value, default in self.ordered_non_default():
            prtf("{}={} (default={})".format(attr.upper(), value, default))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in self.ordered_params():
            text += "|{}|{}|  \n".format(attr, value)

        return text

    def non_default_as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|default|  \n|-|-|-|  \n"
        for attr, value, default in self.ordered_non_default():
            text += "|{}|{}|{}|  \n".format(attr, value, default)

        return text

    def build_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--local_rank', type=int, default=0)

        parser.add_argument('--seed', type=int, default=0)

        parser.add_argument('--lr-multiplier', type=float, default=1.0)
        parser.add_argument('--no-perceptual-loss', action='store_true')
        parser.add_argument('--batch-size', type=int, default=2048)
        parser.add_argument('--epoch-size', type=int, default=65536)
        parser.add_argument('--validation-size', type=int, default=1024)
        parser.add_argument('--image-count', type=int, default=32)
        parser.add_argument('--image-dim', type=int, default=128)
        parser.add_argument('--rays-per-tri', type=int, default=128)
        parser.add_argument('--samples-per-ray', type=int, default=512)
        parser.add_argument('--no-cudnn-benchmark', action='store_true')
        parser.add_argument('--no-fused-adam', action='store_true')
        parser.add_argument('--opt-level', default='O0')
        parser.add_argument('--epochs', type=int, default=300)
        parser.add_argument('--amp-verbosity', type=int, default=0)
        parser.add_argument('--save-model-every', type=int, default=5)
        parser.add_argument(
            '--display-freq',
            type=int,
            default=4096,
            help='number of samples per display print out and tensorboard save'
        )
        parser.add_argument('--set-lr-freq',
                            type=int,
                            default=4096,
                            help='number of samples per setting optimizer lr')
        parser.add_argument('--show-model-info', action='store_true')
        parser.add_argument('--name', required=True)

        parser.add_argument('--use-attention', action='store_true')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))
