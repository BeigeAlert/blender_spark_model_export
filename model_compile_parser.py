# 1453703568
# Helps tokenize the model_compile text block
# Written by Trevor "BeigeAlert" Harris


class Line:
    def __init__(self, text, line_number):
        self.text = text
        self.line_number = line_number


class TokenizedReader:
    def __init__(self, text):
        temp_lines = text.split('\n')
        temp_line_items = [None] * len(temp_lines)
        for i in range(0, len(temp_line_items)):
            temp_line = Line(temp_lines[i].strip(), i)
            temp_line_items[i] = temp_line
        self.lines = [s for s in temp_line_items if s.text != '']
        self.pos = 0  # character position within the current line
        self.line = 0  # current line number

    def get_token(self):
        if self.line >= len(self.lines):
            return None
        text = self.lines[self.line].text
        if self.pos >= len(text):
            self.line += 1
            self.pos = 0
            return self.get_token()
        line_num = str(self.lines[self.line].line_number)  # string of the line number in the source
        pos = self.pos  # convenience
        
        if text[pos] == '"':  # token is a string literal, scan for closing quotes
            start_pos = pos
            pos += 1
            while text[pos] != '"':
                pos += 1
                if pos >= len(text):
                    raise Exception("Unclosed string at line " + line_num + ". Exiting.")
            pos += 1  # to include the "
            self.pos = pos
            return text[start_pos + 1:pos - 1]  # potentially an empty string if they write "", but I'm not too worried

        elif text[pos].isspace():  # somehow ran into some whitespace at the beginning... shouldn't happen...
            while text[pos].isspace():
                pos += 1
                if pos >= len(text):  # whitespace took us to the end of the line, go to the next line, and recurse
                    self.line += 1
                    return self.get_token()
            # hit the end of the whitespace, returning the next token
            self.pos = pos
            return self.get_token()

        else:  # regular characters found
            start_pos = pos
            while (not text[pos].isspace()) and (not text[pos] == '"'):
                pos += 1
                if pos >= len(text):  # regular characters right up to the endl
                    break
                if text[pos - 1:pos + 1] == '//':  # comments skip the rest of the line
                    self.line += 1
                    self.pos = 0
                    if (pos - 1) - start_pos > 0:
                        return text[start_pos:pos - 1]
                    else:
                        return self.get_token()
                if text[pos - 1:pos + 1] == '/*':  # keep skipping characters until we find '*/'
                    return_token = None
                    if pos - 1 - start_pos > 0:
                        return_token = text[start_pos:pos - 1]  # token ends at '/*', but we still need to run to
                        # the end of the comment before returning
                    while True:
                        pos += 1
                        if pos >= len(text):  # end of line hit, comment spans multiple lines
                            self.line += 1
                            self.pos = 1
                            text = self.lines[self.line].text
                            while len(text) < 2:
                                self.line += 1
                                text = self.lines[self.line].text
                        if text[pos - 1:pos + 1] == '*/':
                            self.pos = pos + 1
                            if return_token is not None:
                                return return_token
                            else:
                                return self.get_token()
            self.pos = pos
            return text[start_pos:pos]
    
    def peek_token(self):
        temp_pos = self.pos
        temp_line = self.line
        token = self.get_token()
        self.pos = temp_pos
        self.line = temp_line
        return token
    
    def has_token(self):
        if self.peek_token() is None:
            return False
        return True
    
    def get_line(self):
        return self.lines[self.line].line_number + 1

