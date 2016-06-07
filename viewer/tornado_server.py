import tornado.ioloop
import tornado.web
from benchmark import BenchmarkDatabase

class MainHandler(tornado.web.RequestHandler):
    def get(self):
	self.write(self.request.uri)

class DataRequestHandler(tornado.web.RequestHandler):
    def get(self):
	project = self.request.uri.split("/")
        path = "/home/openmdao/webapps/benchmark_data_server/"
        project_name = project[1] 
        spec = project[2]
        dbfullpath = path + project_name
        db = BenchmarkDatabase(dbfullpath)
       
        data = {}
        for row in db.cursor.execute("SELECT * FROM BenchmarkData WHERE Spec=? and Status=='OK' ORDER BY DateTime", (spec,)):
            data.setdefault('timestamp', []).append(row[0])
            data.setdefault('status', []).append(row[2])
            data.setdefault('elapsed', []).append(row[3])
            data.setdefault('memory', []).append(row[4])
            data.setdefault('LoadAvg1m', []).append(row[5])
            data.setdefault('LoadAvg5m', []).append(row[6])
            data.setdefault('LoadAvg15m', []).append(row[7])

        if not data:
             print("No data to plot for %s" % spec)
        else: print data 

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/\w+/\w+", DataRequestHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(18309)
    tornado.ioloop.IOLoop.current().start()
