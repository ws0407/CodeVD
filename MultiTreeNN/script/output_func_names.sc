@main def exec(cpgFile: String, outFile: String) = {
  importCpg(cpgFile)
  var func_name = cpg.method.filter(_.isExternal == false).name.l
  var x = func_name.toString
  x |> outFile
}