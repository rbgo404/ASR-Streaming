from whisper_online import *
import time
import base64
import os
import os

class InferlessPythonModel:

  def initialize(self):
      asr = FasterWhisperASR("en", "large-v2")
      self.online = OnlineASRProcessor(asr)
      self.audio_path = "temp.wav"

  def base64_to_mp3(self, base64_data, output_file_path):
      # Convert base64 audio data to mp3 file
      mp3_data = base64.b64decode(base64_data)
      with open(output_file_path, "wb") as mp3_file:
          mp3_file.write(mp3_data)

  def infer(self,inputs,stream_output_handler):
      audio_data = inputs["audio_base64"]
      self.base64_to_mp3(audio_data, self.audio_path)

      chunk_duration = 2
      beg = 0.0
      SAMPLING_RATE = 16000
      duration = len(load_audio(self.audio_path))/SAMPLING_RATE
      while True:
          end = beg + chunk_duration
          if end > duration:
              end = duration
          a = load_audio_chunk(self.audio_path, beg, end)
          self.online.insert_audio_chunk(a)
          
          output = self.online.process_iter()
          output_dict = {}
          output_dict["OUT"] = output[-1]
          stream_output_handler.send_streamed_output(output_dict)

          if end >= duration:
              break

          beg = end
      output = self.online.finish()
      self.online.init()
      stream_output_handler.finalise_streamed_output()
      os.remove(self.audio_path)

  def finalize(self):
    self.online = None
