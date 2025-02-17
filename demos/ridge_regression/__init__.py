import termcolor
class FormattedFileWriter:
    """A class for writing formatted output to a file with optional ANSI formatting.
    
    Attributes:
        file_path (str): Path to the file where the output will be written.
        use_formatting (bool): Indicates whether formatting should be applied 
                               based on file extension.
    """

    def __init__(self, file_path: str):
        """Initializes the FormattedFileWriter with the file path.

        Args:
            file_path (str): The path to the file where the output will be written.
        """
        if file_path is None:
            raise ValueError("file_path cannot be None")
        self.file_path = file_path
        self.use_formatting = file_path.endswith(('.ans', '.ansi'))

    def write(self, text: str, color: str = None, on_color: str = None, attrs: list = None):
        """Writes the text to the file, optionally with formatting.

        Args:
            text (str): The text to write.
            color (str, optional): Foreground color (e.g., 'red', 'green'). Defaults to None.
            on_color (str, optional): Background color (e.g., 'on_red', 'on_green'). Defaults to None.
            attrs (list, optional): Additional text attributes (e.g., ['bold', 'underline']). 
                                    Defaults to None.
        """
        if self.use_formatting:
            formatted_text = termcolor.colored(text, color=color, on_color=on_color, attrs=attrs)
        else:
            formatted_text = text
        
        # Open the file in append mode, creating it if it doesn't exist
        with open(self.file_path, 'a') as file:
            file.write(formatted_text + '\n')