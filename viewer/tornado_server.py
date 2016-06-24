import os
from datetime import datetime

import tornado.ioloop
import tornado.web
from benchmark import BenchmarkDatabase

#database_dir = os.path.abspath(os.path.dirname(__file__))
database_dir = "/home/openmdao/webapps/benchmark_data_server/"


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
        for row in db.cursor.execute('SELECT * FROM BenchmarkData WHERE Spec=? and Status=="OK" ORDER BY DateTime ASC', (spec,)):
            data.setdefault("timestamp", []).append(row[0])
            data.setdefault("status", []).append(row[2])
            data.setdefault("elapsed", []).append(row[3])
            data.setdefault("memory", []).append(row[4])
            data.setdefault("LoadAvg1m", []).append(row[5])
            data.setdefault("LoadAvg5m", []).append(row[6])
            data.setdefault("LoadAvg15m", []).append(row[7])

        commits = []

        for timestamp in data['timestamp']:
            tmp_list = []
            for row in db.cursor.execute('SELECT * FROM Commits WHERE DateTime==? ORDER BY DateTime', (timestamp,)):
                # row[0] is timestamp, row[1] is trigger, row[2] is commit
                name = row[1].rsplit('/', 1)[1]
                link = (row[1]+'/commit/'+row[2]).strip().replace(':', '/').replace('git@', 'http://')
                tmp_list.append((name, link))
            commits.append(tmp_list)

        data['commits'] = commits

        from pprint import pprint
        pprint(data['timestamp'])
        pprint(data['commits'])

        # NOTE: this dictionary is probably not the best data structure to use here
        #       (depending on the ultimate means of rendering it to HTML)

        if not data:
            print("No data to plot for %s" % spec)
            return "No data for %s %s " % (project, spec)
        else:
            bench_title = "Benchmark Results for " + spec

            def date(timestamp):
                return str(datetime.fromtimestamp(timestamp))

            self.render("template.html", title=bench_title, items=data, date=date)


def make_app():
    return tornado.web.Application([
        (r"/(.*)/(.*)", SpecHandler),
        (r"/(.*)",      ProjectHandler),
        (r"/",          MainHandler),
        (r"/*.js", tornado.web.StaticFileHandler, dict(path='.'))
    ], debug=True)

if __name__ == "__main__":
    app = make_app()
    app.listen(18309)
    tornado.ioloop.IOLoop.current().start()
