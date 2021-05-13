import shutil
import glob
import pathlib

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as build_ext_orig


class CMakeExtension(Extension):
    def __init__(self, name):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[])


class build_ext(build_ext_orig):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        # this is quite hacky...

        build_arg_mode = ['--debug'] if self.debug else ['--release']
        gen_args = ['--skip-build'] if self.dry_run else []

        # example of build args
        self.spawn(['cmake_cli'] +
                   ['build', '--target', 'neural_render_generate_data'] +
                   build_arg_mode + gen_args)

        # these dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        extdir.parent.mkdir(parents=True, exist_ok=True)
        file = 'build/{}/{}'.format('debug' if self.debug else 'release',
                                    extdir.name)

        shutil.copyfile(file, extdir.absolute())


setup(
    name='neural_render_generate_data',
    ext_modules=[CMakeExtension('neural_render_generate_data')],
    cmdclass={
        'build_ext': build_ext,
    },
)