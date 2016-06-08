import os
from datetime import datetime

import tornado.ioloop
import tornado.web
from benchmark import BenchmarkDatabase

database_dir = os.path.abspath(os.path.dirname(__file__))
# database_dir = "/home/openmdao/webapps/benchmark_data_server/"


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        print "==> MainHandler:", self.request.uri
        self.write(self.request.uri)

class ProjectHandler(tornado.web.RequestHandler):
    def get(self, project):
        print "==> ProjectHandler:", project
        self.write("Project: "+project)

class SpecHandler(tornado.web.RequestHandler):
    def get(self, project, spec):
        print "==> SpecHandler:", project, spec
        print os.path.join(database_dir, project)
        db = BenchmarkDatabase(os.path.join(database_dir, project))

        data = {}
        for row in db.cursor.execute('SELECT * FROM BenchmarkData WHERE Spec=? and Status=="OK" ORDER BY DateTime', (spec,)):
            data.setdefault("timestamp", []).append(row[0])
            data.setdefault("status", []).append(row[2])
            data.setdefault("elapsed", []).append(row[3])
            data.setdefault("memory", []).append(row[4])
            data.setdefault("LoadAvg1m", []).append(row[5])
            data.setdefault("LoadAvg5m", []).append(row[6])
            data.setdefault("LoadAvg15m", []).append(row[7])

        # NOTE: this dictionary is probably not the best data structure to use here
        #       (depending on the ultimate means of rendering it to HTML)

        if not data:
            print("No data to plot for %s" % spec)
            response = "No data for %s %s " % (project, spec)
        else:
            response = "{:s}: {:s}\n".format(project, spec)
            line_fmt = "{:s} {:4s} {:7.2f} {:7.2f} {:7.2f} {:7.2f} {:7.2f}\n"
            for i in range(len(data["timestamp"])):
                response += line_fmt.format(
                    str(datetime.fromtimestamp(data["timestamp"][i])),
                    data["status"][i],
                    data["elapsed"][i],
                    data["memory"][i],
                    data["LoadAvg1m"][i],
                    data["LoadAvg5m"][i],
                    data["LoadAvg15m"][i]
                )

        self.write(response)

def make_app():
    return tornado.web.Application([
        (r"/(.*)/(.*)", SpecHandler),
        (r"/(.*)",      ProjectHandler),
        (r"/",          MainHandler),
    ], debug=True)

if __name__ == "__main__":
    app = make_app()
    app.listen(18309)
    tornado.ioloop.IOLoop.current().start()
