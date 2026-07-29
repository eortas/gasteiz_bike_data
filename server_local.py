import http.server
import socketserver
import os
import sys

# Servidor local de desarrollo para probar la web de Mugibike en local
PORT = 8000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, BASE_DIR)
from api.calcular import handler as CalcularHandler

class LocalDevHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/api/calcular'):
            CalcularHandler.do_GET(self)
        else:
            super().do_GET()

if __name__ == '__main__':
    os.chdir(BASE_DIR)
    print(f"🚀 Servidor local de Mugibike activo en: http://localhost:{PORT}")
    print("Abre tu navegador e introduce la dirección anterior. Pulsa Ctrl+C para detenerlo.")
    try:
        with socketserver.TCPServer(("", PORT), LocalDevHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
