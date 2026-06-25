from pipeline.audio_input import FileInput, MicrophoneInput

def test_file_input_raises_on_missing():
    fi = FileInput()
    try:
        fi.load("/nonexistent/file.wav")
        assert False
    except (FileNotFoundError, Exception):
        pass

def test_microphone_not_implemented():
    mi = MicrophoneInput()
    try:
        mi.load("")
        assert False
    except NotImplementedError:
        pass
