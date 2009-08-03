import os

from tools.config_dict import ConfigDict

class IOContext(ConfigDict):

    def __init__(self, config_file=None, extra_options={}):
        ConfigDict.__init__(self)
        if config_file is None:
            config_file = os.path.join(os.path.dirname(
                                       os.path.abspath(__file__)),
                                       'default_params.conf')
        self.fromfile(config_file)
        self.update(extra_options)
        if not 'output_dir' in self:
            self['output_dir'] = os.path.join(os.getcwd(), 'output')
        self['cachedir'] = os.path.join(self['output_dir'], 'cache')
        for key, value in self.iteritems():
            if key.endswith('dir'):
                self[key] = os.path.expanduser(value)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


    def outpath(self, filename):
        """ Return a path to the file name in the output directory.
        """
        return os.path.join(self.output_dir, filename)


