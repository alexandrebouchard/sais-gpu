//// Generate cross products that keep track of both variable names and values

def crossProduct(Map<String, List<?>> mapOfLists, boolean dryRun) {
    if (dryRun) {
        def result = [:]
        for (key in mapOfLists.keySet()) {
            def list = mapOfLists[key]
            result[key] = list.first()
        }
        return Channel.of(result)
    } else 
        crossProduct(mapOfLists)
}

def crossProduct(Map<String, List<?>> mapOfLists) {
    def keys = new ArrayList(mapOfLists.keySet())
    def list = _crossProduct(mapOfLists, keys)
    return Channel.fromList(list)
}

def _crossProduct(mapOfLists, keys) {
    if (keys.isEmpty()) {
        return [[:]]
    }
    def key = keys.remove(keys.size() - 1)
    def result = []
    for (recursiveMap : _crossProduct(mapOfLists, keys))
        for (value : mapOfLists.get(key)) {
            def copy = new LinkedHashMap(recursiveMap)
            copy[key] = value 
            result.add(copy)
        }
    return result
}


//// Collect CSVs from many execs

def collectCSVs(inputChannel) { collectCSVsProcess(inputChannel.map{toJson(it)}.toList()) }

def toJson(tuple) {
    def args = tuple[0]
    def path = tuple[1]
    return groovy.json.JsonOutput.toJson([
        path: path.toString(), 
        args: args
    ])
}

process collectCSVsProcess {
    debug true
    cache true
    scratch false
    input: 
        // we use file not path, as we want the json strings to be dumped into files
        file jsonObjects
    output:
        path aggregated
    publishDir { deliverables(workflow) }, mode: 'copy', overwrite: true
    """
    aggregate
    """
}

def deliverables(workflow) { 'deliverables/' + workflow.scriptName.replace('.nf','') + "_" + workflow.start }

