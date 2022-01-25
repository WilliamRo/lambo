from lambo.zebra.io.inflow import Inflow
from lambo.zebra.io.drivers.flir_api import FLIRCamDev



class FLIRFetcher(Inflow):

  def __init__(self, max_len=20):
    super(FLIRFetcher, self).__init__(max_len)

    self.camera = FLIRCamDev()


  def _init(self):
    self.camera.start()
    self.camera.set_buffer_count(1)


  def _loop(self):
    im = self.camera.read().GetNDArray()
    self.append_to_buffer(im)


  def _finalize(self):
    self.camera.stop()
    self.camera.close()
