"""
A Python file to open the camera and recognize the hand gesture
"""
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.uix.label import Label
import time
from workers.evaluate import ClassificationModel

text = 'Loading'
Builder.load_string("""
<CameraClick>:
    orientation: 'vertical'
    Camera:
        id: camera
        resolution: (480, 480)
        play: False
        
    BoxLayout:
        orientation: 'horizontal'
        size_hint_y: None
        height: '48dp'
        ToggleButton:
            text: 'Play'
            on_press: camera.play = not camera.play
            size_hint_y: None
            height: '48dp'
        Button:
            text: 'Capture'
            size_hint_y: None
            height: '48dp'
            on_press: root.capture()
""")

model_path = 'models/small_model'


class CameraClick(BoxLayout):
    """A class that contains a function which displays the meaning of the hand gestures"""

    def capture(self) -> None:
        """ Function to capture the images and give them the names
        according to their captured time and date.
        """
        camera = self.ids['camera']
        time_str = time.strftime("%Y%m%d_%H%M%S")
        img_name = "IMG_{}.png".format(time_str)
        camera.export_to_png(img_name)
        print("Captured")
        cl = ClassificationModel(model_path)
        result = cl.run_from_file_path(img_name)
        message = 'Letter: ' + result[0]
        self.popup_control(message)

    def popup_control(self, final_message: str):
        """A function that controls the pop-up message"""
        total_widgets = BoxLayout(orientation='vertical')
        total_widgets.add_widget(Label(text=final_message))
        popup = Popup(title='Result', content=total_widgets, size_hint=(None, None),
                      size=(300, 200), auto_dismiss=True)
        popup.open()


class TestCamera(App):
    """A class that opens the window with a camera"""

    def build(self):
        return CameraClick()


TestCamera().run()
